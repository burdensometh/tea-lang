"""Tea's public API.

Two equivalent entry points are supported:

1. Explicit sessions (:class:`Session`). Each session owns its own data
   snapshot, variable definitions, study design, assumptions, alpha and an
   independent :class:`~tea.z3_solver.solver.SolverContext`. Two sessions can
   be driven interleaved in the same process; their analyses never share
   state, and produced :class:`~tea.runtimeDataStructures.varData.VarData`,
   applied properties and :class:`~tea.runtimeDataStructures.resultData.ResultData`
   stay bound to the session that created them.

2. Module-level functions (``tea.data``, ``tea.hypothesize``, ...). These are
   backward-compatible thin wrappers that delegate to a process-wide default
   session, so existing single-analysis programs keep working unchanged.
"""
from tea.logging.tea_logger import TeaLogger
from tea.helpers.study_type_determiner import StudyTypeDeterminer
import pandas as pd

from tea.runtimeDataStructures.resultData import ResultData
from tea.runtimeDataStructures.dataset import Dataset
from tea.z3_solver.solver import SolverContext

from .build import (load_data, load_data_from_url, const,
                    ordinal, isordinal,
                    nominal, isnominal,
                    ratio, isratio,
                    interval, isinterval, isnumeric,
                    select, compare, relate, predict,
                    get_var_from_list
                    )
from .vardata_factory import VarDataFactory
from .hypotheses import _RelationshipSpec
from .ast import Relate, Relationship, PositiveRelationship, NegativeRelationship
import tea.helpers
import tea.runtimeDataStructures
import tea.z3_solver
from tea.helpers.constants.default_values import DEFAULT_ALPHA_PARAMETER

from typing import Dict, Optional
from .global_vals import *
from pathlib import Path

# For variables dictionary
var_name = 'name'
var_dtype = 'data type'
var_categories = 'categories'
var_drange = 'range'


class Session:
    """An isolated analysis session.

    Holds everything one analysis (or a family of analyses over the same data
    snapshot) needs: the dataset snapshot, declared variables, study design,
    assumptions, alpha, solver mode and its own solver context. Results
    returned by :meth:`hypothesize` are bound to the session and are never
    rewritten by later changes to the session's data or definitions.
    """

    def __init__(self, name: Optional[str] = None):
        # Human-readable handle, useful when juggling several sessions.
        self.name = name

        # Data snapshot (materialized lazily on the first hypothesize call).
        self.dataset_path = ''
        self.dataset_id = None
        self.dataset_obj: Optional[Dataset] = None

        # Declared variables, study design and assumptions.
        self.vars_objs = []
        self.study_design = None
        self.assumptions: Dict[str, str] = {}
        self.alpha = DEFAULT_ALPHA_PARAMETER

        # Solver behavior for analyses run through this session.
        self.MODE = 'strict'

        # Results produced by this session, keyed by hypothesis call order.
        # Kept (instead of mutated) so later data/definition changes cannot
        # rewrite earlier results.
        self.all_results = {}
        self._result_counter = 0

        # A fresh determiner/factory per session keeps even these stateless
        # collaborators from being shared implicitly across analyses.
        self.study_type_determiner = StudyTypeDeterminer()

    # ------------------------------------------------------------------
    # Configuration steps (mirror the legacy module-level functions)
    # ------------------------------------------------------------------
    def data(self, file, key=None):
        # Require that the path to the data must be a string or a Path object
        assert isinstance(file, (str, Path, pd.DataFrame))
        self.dataset_path = file
        self.dataset_id = key
        # A new data declaration invalidates any previously materialized snapshot.
        self.dataset_obj = None

    def define_variables(self, vars: Dict[str, str]):
        # reset the variables
        self.vars_objs = []

        for var in vars:
            name = var['name']
            if (var[var_dtype] == 'nominal'):
                categories = var[var_categories]
                v_obj = nominal(name, categories)
            elif (var[var_dtype] == 'ordinal'):
                categories = var[var_categories]
                v_obj = ordinal(name, categories)
            elif (var[var_dtype] == 'interval'):
                drange = None
                if var_drange in var:
                    drange = var[var_drange]
                v_obj = interval(name, drange)
            else:
                assert (var[var_dtype] == 'ratio')
                drange = var[var_drange] if var_drange in var else None
                v_obj = ratio(name, drange)

            self.vars_objs.append(v_obj)

    def define_study_design(self, design: Dict[str, str]):
        # Check that variables are only assigned EITHER between OR within but NOT BOTH:
        btw_vars = design[btw_subj] if btw_subj in design else None
        within_vars = design[within_subj] if within_subj in design else None

        if btw_vars:
            for b in btw_vars:
                if within_vars:
                    for w in within_vars:
                        if b == w:
                            raise ValueError(
                                f"{b} CANNOT be a between subjects variable AND a within subjects variable. Can only be one or the other.")

        self.study_design = design

    def assume(self, user_assumptions: Dict[str, str], mode=None):
        tea_logger = TeaLogger.get_logger()

        if alpha_keywords[0] in user_assumptions:
            if alpha_keywords[1] in user_assumptions:
                assert (float(user_assumptions[alpha_keywords[0]]) == float(user_assumptions[alpha_keywords[1]]))

        for keyword in alpha_keywords:
            if keyword in user_assumptions:
                self.alpha = float(user_assumptions[keyword])

        # Defensive copy: callers mutating their dict afterwards must not
        # change what this session assumed.
        self.assumptions = dict(user_assumptions)
        self.assumptions[alpha_keywords[1]] = self.alpha

        # Set MODE for dealing with assumptions
        if mode and mode == 'relaxed':
            self.MODE = mode
            tea_logger.log_info(f"\nRunning under {self.MODE.upper()} mode.\n")
            tea_logger.log_info(
                f"This means that user assertions will be checked. Should they fail, Tea will issue a warning but proceed as if user's assertions were true.")
        else:
            assert (mode is None or mode == 'strict')
            self.MODE = 'strict'
            tea_logger.log_info(f"\nRunning under {self.MODE.upper()} mode.\n")
            tea_logger.log_info(f"This means that user assertions will be checked. Should they fail, Tea will override user assertions.\n")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _materialize_dataset(self) -> Dataset:
        """Load (once) the immutable data snapshot for this session."""
        if self.dataset_obj is None:
            source = self.dataset_path
            # Copy DataFrames so external mutation cannot alter the snapshot.
            if isinstance(source, pd.DataFrame):
                source = source.copy()
            self.dataset_obj = load_data(source, self.vars_objs, self.dataset_id)
        return self.dataset_obj

    def hypothesize(self, vars: list, prediction: list = None):
        if isinstance(self.dataset_path, (str, Path)):
            assert (self.dataset_path)  # assert not an empty string
        elif isinstance(self.dataset_path, pd.DataFrame):
            pass
        else:
            raise ValueError(f"dataset_path must be DataFrame, str, or Path. Not: {type(self.dataset_path)}")
        assert (self.vars_objs)
        assert (self.study_design)

        # Snapshot the data: every analysis in this session reads the same
        # Dataset instance, regardless of later changes to the source file or
        # the caller's DataFrame.
        dataset_obj = self._materialize_dataset()

        v_objs = []
        for v in vars:
            v_objs.append(get_var_from_list(v, self.vars_objs))  # may want to use Dataset instance method instead

        # Create and get back handle to AST node
        if _is_structured(prediction):
            relationship = _build_structured_relate(v_objs, prediction)
        else:
            relationship = relate(v_objs, prediction)
        num_comparisons = len(relationship.predictions) if len(relationship.predictions) > 0 else 1  # use for multiple comparison correction

        # A fresh, session-owned context for every analysis. Its registries
        # start empty and are simply discarded afterwards, so neither a failed
        # run nor a concurrent/ interleaved analysis can leave applied
        # properties behind that influence the next test selection.
        context = SolverContext(mode=self.MODE, alpha=self.alpha, session=self)
        context.set_mode(self.MODE)

        vardata_factory = VarDataFactory(self.study_type_determiner)
        result = vardata_factory.create_vardata(dataset_obj, relationship,
                                                self.assumptions, self.study_design,
                                                context=context)

        if dataset_obj.data.empty:
            # If the the result is a dictionary of test names, which should only happen if the data is empty
            assert (isinstance(result, list))
            assert (dataset_obj.has_empty_data())
            return result

        # Else
        assert (isinstance(result, ResultData))
        # Make multiple comparison correction
        result.bonferroni_correction(num_comparisons)

        # Freeze the result against this session; never overwrite it later.
        self._result_counter += 1
        self.all_results[self._result_counter] = result

        print(f"\n{result}")
        return result

    def divine_properties(self, vars: list, tests: list):
        v_objs = []
        for v in vars:
            v_objs.append(get_var_from_list(v, self.vars_objs))  # may want to use Dataset instance method instead

        relationship = relate(v_objs)

        # What kind of study are we analyzing?
        study_type = self.study_type_determiner.determine_study_type(vars, self.study_design)

        from tea.runtimeDataStructures.bivariateData import BivariateData
        from tea.runtimeDataStructures.multivariateData import MultivariateData

        # Do we have a Bivariate analysis?
        if len(vars) == 2:
            combined_data = BivariateData(v_objs, study_type, alpha=float(self.assumptions['alpha']))
        else:  # Do we have a Multivariate analysis?
            combined_data = MultivariateData(v_objs, study_type, alpha=float(self.assumptions['alpha']))

        # Run property synthesis on this session's own context.
        context = SolverContext(mode=self.MODE, alpha=self.alpha, session=self)
        context.construct_all_tests(combined_data)
        test_to_properties, test_to_broken_properties = context.which_props(tests, vars)

        all_properties_are_satisfied = True
        for val in test_to_broken_properties.values():
            if val:
                all_properties_are_satisfied = False
                break

        if all_properties_are_satisfied:
            print(f"\nProperties for {tests[0]} and {tests[1]} are complementary.")
        else:
            print(f"\nProperties for {tests[0]} and {tests[1]} conflict.")

        import pprint
        pp = pprint.PrettyPrinter()

        print("\nProperties:")
        pp.pprint(test_to_properties)
        print("\nProperties that could not be satisfied:")
        pp.pprint(test_to_broken_properties)


def _is_structured(prediction):
    """Return True if prediction uses the structured API (not string-based)."""
    return prediction is not None and len(prediction) > 0 and not isinstance(prediction[0], str)


def _build_structured_relate(v_objs, predictions):
    """Build a Relate node from structured prediction objects."""
    resolved = []
    for pred in predictions:
        if isinstance(pred, list):
            # From compare() — already contains AST nodes, wrap for categorical shape
            resolved.append(pred)
        elif isinstance(pred, _RelationshipSpec):
            var1 = get_var_from_list(pred.var1_name, v_objs)
            var2 = get_var_from_list(pred.var2_name, v_objs)
            if var1 is None:
                raise ValueError(f"Variable {pred.var1_name!r} not found")
            if var2 is None:
                raise ValueError(f"Variable {pred.var2_name!r} not found")
            if pred.direction == 'positive':
                resolved.append(PositiveRelationship(Relationship(var1), Relationship(var2)))
            else:
                resolved.append(NegativeRelationship(Relationship(var1), Relationship(var2)))
        else:
            raise ValueError(f"Unsupported prediction object: {pred!r}")
    return Relate(v_objs, resolved)


# ---------------------------------------------------------------------------
# Default session and backward-compatible module-level API.
# These functions are pure delegation: isolation is provided by Session and
# SolverContext, not by locking shared global dictionaries.
# ---------------------------------------------------------------------------
_default_session = Session(name='default')


def _sync_module_globals(session: Session):
    """Mirror the default session's state onto the legacy module globals."""
    globals()['dataset_path'] = session.dataset_path
    globals()['dataset_obj'] = session.dataset_obj
    globals()['dataset_id'] = session.dataset_id
    globals()['vars_objs'] = session.vars_objs
    globals()['study_design'] = session.study_design
    globals()['assumptions'] = session.assumptions
    globals()['alpha'] = session.alpha
    globals()['all_results'] = session.all_results
    globals()['MODE'] = session.MODE
    globals()['study_type_determiner'] = session.study_type_determiner


# Legacy module globals, kept in sync with the default session so code and
# tests that import them observe the same values as before.
dataset_path = _default_session.dataset_path
dataset_obj = _default_session.dataset_obj
dataset_id = _default_session.dataset_id
vars_objs = _default_session.vars_objs
study_design = _default_session.study_design
assumptions = _default_session.assumptions
alpha = _default_session.alpha
all_results = _default_session.all_results
MODE = _default_session.MODE
study_type_determiner = _default_session.study_type_determiner


def create_session(name: Optional[str] = None) -> Session:
    """Create a new, fully isolated analysis session."""
    return Session(name=name)


# For testing purposes
def download_data(url, file_name):
    return load_data_from_url(url, file_name)


# @sets default session's dataset_path and dataset_obj
def data(file, key=None):
    _default_session.data(file, key)
    _sync_module_globals(_default_session)


def define_variables(vars: Dict[str, str]):
    _default_session.define_variables(vars)
    _sync_module_globals(_default_session)


def define_study_design(design: Dict[str, str]):
    _default_session.define_study_design(design)
    _sync_module_globals(_default_session)


def assume(user_assumptions: Dict[str, str], mode=None):
    _default_session.assume(user_assumptions, mode)
    _sync_module_globals(_default_session)


def hypothesize(vars: list, prediction: list = None):
    result = _default_session.hypothesize(vars, prediction)
    _sync_module_globals(_default_session)
    return result


# TODO: Add relate and compare methods

# @param vars that user would like to relate
# @param stats_tests contains all the tests that the user would like
# @return properties that must be true in order to satisfy
# as many of the tests as possible
def divine_properties(vars: list, tests: list):
    _default_session.divine_properties(vars, tests)
    _sync_module_globals(_default_session)
