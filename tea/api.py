from tea.logging.tea_logger import TeaLogger
from tea.helpers.study_type_determiner import StudyTypeDeterminer
import pandas as pd

from tea.runtimeDataStructures.resultData import ResultData

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
from .runtimeDataStructures.covariateJoin import (
    CovariateJoinSpec,
    JoinEvidence,
    validate_attribute_schema,
)
from .runtimeDataStructures.dataset import Dataset
import tea.helpers
import tea.runtimeDataStructures
import tea.z3_solver
from tea.helpers.constants.default_values import DEFAULT_ALPHA_PARAMETER
from tea.z3_solver.solver import set_mode

from typing import Dict, Optional
from .global_vals import *
from pathlib import Path

# Set at start of programs
# Used across functions
dataset_path = ''
dataset_obj = None
dataset_id = None
vars_objs = []
study_design = None

# External (participant-scoped) covariate history.  Declared with
# attach_covariates(); consumed by hypothesize() to build the same joined
# snapshot that variable/attribute validation sees.
covariate_history = None
covariate_spec = None
covariate_evidence = None

# For variables dictionary
var_name = 'name'
var_dtype = 'data type'
var_categories = 'categories'
var_drange = 'range'

# Assumptions
# Stats properties
assumptions = {}
alpha = DEFAULT_ALPHA_PARAMETER

all_results = {}  # Used for multiple comparison correction

# For solver
MODE = 'strict'

study_type_determiner = StudyTypeDeterminer()


# For testing purposes
def download_data(url, file_name):
    return load_data_from_url(url, file_name)


# @sets global dataset_path and dataaset_obj (of type Dataset)
def data(file, key=None):
    global dataset_path, dataset_obj, dataset_id
    global covariate_history, covariate_spec, covariate_evidence

    # Require that the path to the data must be a string or a Path object
    assert isinstance(file, (str, Path, pd.DataFrame))
    dataset_path = file
    dataset_id = key

    # A new dataset invalidates any previously declared covariate history.
    dataset_obj = None
    covariate_history = None
    covariate_spec = None
    covariate_evidence = None


def attach_covariates(history, spec: CovariateJoinSpec):
    """Declare a participant-scoped history table to join onto the dataset.

    The join happens when ``hypothesize`` runs, so the variable schema,
    attribute validation and the analysis itself all read the same post-join
    snapshot.  ``spec`` declares the event-time column, the effective-time
    column, the optional expiry column, the participant key and the policies
    for overlaps, duplicates and missing matches; see
    :class:`tea.runtimeDataStructures.covariateJoin.CovariateJoinSpec`.

    ``history`` may be a DataFrame, a CSV path, or another Dataset.  Attributes
    it contributes must also be named in ``define_variables``.
    """
    global covariate_history, covariate_spec, covariate_evidence

    if not isinstance(spec, CovariateJoinSpec):
        raise ValueError(
            "attach_covariates expects a CovariateJoinSpec (declare the event "
            f"time, effective time and participant key). Received: {type(spec)!r}"
        )
    if not isinstance(history, (pd.DataFrame, str, Path, Dataset)):
        raise ValueError(
            "history must be a DataFrame, CSV path, or Dataset. "
            f"Received: {type(history)!r}"
        )

    covariate_history = history
    covariate_spec = spec
    covariate_evidence = None

    # Declaring a new history invalidates any previously built analysis object.
    dataset_obj = None


def get_covariate_evidence() -> Optional[JoinEvidence]:
    """Return the join audit trail from the most recent hypothesize()."""
    return covariate_evidence


def define_variables(vars: Dict[str, str]):
    global vars_objs

    # reset the variables
    vars_objs = []

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

        vars_objs.append(v_obj)


def define_study_design(design: Dict[str, str]):
    global study_design, dataset_id, uid, alpha
    global btw_subj, within_subj

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

    study_design = design

    # dataset_id = design[uid] if uid in design else None


def assume(user_assumptions: Dict[str, str], mode=None):
    
    tea_logger = TeaLogger.get_logger()
    global alpha, alpha_keywords
    global assumptions
    global MODE

    if alpha_keywords[0] in user_assumptions:
        if alpha_keywords[1] in user_assumptions:
            assert (float(user_assumptions[alpha_keywords[0]]) == float(user_assumptions[alpha_keywords[1]]))

    for keyword in alpha_keywords:
        if keyword in user_assumptions:
            alpha = float(user_assumptions[keyword])

    assumptions = user_assumptions
    assumptions[alpha_keywords[1]] = alpha

    # Set MODE for dealing with assumptions
    if mode and mode == 'relaxed':
        MODE = mode
        tea_logger.log_info(f"\nRunning under {MODE.upper()} mode.\n")
        tea_logger.log_info(
            f"This means that user assertions will be checked. Should they fail, Tea will issue a warning but proceed as if user's assertions were true.")
    else:
        assert (mode == None or mode == 'strict')
        MODE = 'strict'
        tea_logger.log_info(f"\nRunning under {MODE.upper()} mode.\n")
        tea_logger.log_info(f"This means that user assertions will be checked. Should they fail, Tea will override user assertions.\n")


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


def _build_analysis_dataset():
    """Build the single Dataset that schema, validation and tests all consume.

    Without covariate history this is just the loaded data.  With history, the
    join is performed here once, the attached attributes are validated against
    the declared variable schema, and the resulting Dataset carries a snapshot
    tying its data, variables and evidence together.
    """
    global dataset_obj, covariate_history, covariate_spec, covariate_evidence

    dataset_obj = load_data(dataset_path, vars_objs, dataset_id)
    if covariate_history is None or covariate_spec is None:
        return dataset_obj

    dataset_obj = dataset_obj.join_covariates(
        covariate_history, covariate_spec, inplace=True
    )
    covariate_evidence = dataset_obj.evidence

    if covariate_spec.validate:
        # Validate the attached attributes against the declared schema, reading
        # the same post-join frame that hypothesize will analyse.  ``as_of_join``
        # already did this; re-running it here against the snapshot makes the
        # coupling explicit and fails loudly if the snapshot were ever stale.
        attached = list(covariate_evidence.attributes)
        validate_attribute_schema(dataset_obj.data, vars_objs, attached)

    dataset_obj.assert_snapshot_current()
    return dataset_obj


def hypothesize(vars: list, prediction: list = None):
    global dataset_path, vars_objs, study_design, dataset_obj, dataset_id
    global assumptions, all_results
    global MODE

    if isinstance(dataset_path, (str, Path)):
        assert (dataset_path) # assert not an empty string
    elif isinstance(dataset_path, pd.DataFrame):
        # assert not dataset_path.empty
        pass
    else:
        raise ValueError(f"dataset_path must be DataFrame, str, or Path. Not: {type(dataset_path)}")
    assert (vars_objs)
    assert (study_design)

    dataset_obj = _build_analysis_dataset()

    v_objs = []
    for v in vars:
        v_objs.append(get_var_from_list(v, vars_objs))  # may want to use Dataset instance method instead

    # Create and get back handle to AST node
    if _is_structured(prediction):
        relationship = _build_structured_relate(v_objs, prediction)
    else:
        relationship = relate(v_objs, prediction)
    num_comparisons = len(relationship.predictions) if len(relationship.predictions) > 0 else 1 # use for multiple comparison correction

    # Interpret AST node, Returns ResultData object <-- this may need to change
    set_mode(MODE)
    study_type_determiner = StudyTypeDeterminer()
    vardata_factory = VarDataFactory(study_type_determiner)
    result = vardata_factory.create_vardata(dataset_obj, relationship, assumptions, study_design)
    
    if dataset_obj.data.empty:
        # If the the result is a dictionary of test names, which should only happen if the data is empty
        assert(isinstance(result, list))
        assert(dataset_obj.has_empty_data())
        return result

    # Else
    assert(isinstance(result, ResultData))
    # Make multiple comparison correction
    result.bonferroni_correction(num_comparisons)
    
    print(f"\n{result}")
    return result

    # Use assumptions and hypotheses for interpretation/reporting back to user
    # Make result human_readable
    # output = translate(result)

    # Give user output
    # return output


# TODO: Add relate and compare methods

# @param vars that user would like to relate
# @param stats_tests contains all the tests that the user would like
# @return properties that must be true in order to satisfy
# as many of the tests as possible
def divine_properties(vars: list, tests: list):
    global study_design

    v_objs = []
    for v in vars:
        v_objs.append(get_var_from_list(v, vars_objs))  # may want to use Dataset instance method instead

    relationship = relate(v_objs)

    # What kind of study are we analyzing?
    study_type = study_type_determiner.determine_study_type(vars, study_design)

    combined_data = None
    # Do we have a Bivariate analysis?
    if len(vars) == 2:
        combined_data = BivariateData(vars, study_type, alpha=float(assumptions['alpha']))
    else:  # Do we have a Multivariate analysis?
        combined_data = MultivariateData(vars, study_type, alpha=float(assumptions['alpha']))

    # test_to_properties, test_to_broken_properties = which_props(['mannwhitney_u', 'students_t'])
    test_to_properties, test_to_broken_properties = which_props(tests, vars)

    all_properties_are_satisfied = True
    for val in test_to_broken_properties.values():
        if val:
            all_properties_are_satisfied = False
            break

    if all_properties_are_satisfied:
        print(f"\nProperties for {tests[0]} and {tests[1]} are complementary.")
    else:
        print(f"\nProperties for {tests[0]} and {tests[1]} conflict.")

    # print(ps)
    import pprint
    pp = pprint.PrettyPrinter()

    # print("\nProperties for student's t test and Mann Whitney u test are complementary.")
    print("\nProperties:")
    pp.pprint(test_to_properties)
    print("\nProperties that could not be satisfied:")
    pp.pprint(test_to_broken_properties)
