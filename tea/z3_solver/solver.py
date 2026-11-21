"""Z3-backed test synthesis.

All mutable solver state lives on :class:`SolverContext` instances. Every
analysis (each ``hypothesize`` call) creates a fresh context, so two analyses
running in the same process — even interleaved — never share test registries,
applied properties, alpha or solver mode.

The module-level functions kept below are thin backward-compatible shims that
delegate to a legacy context. New code should construct ``SolverContext``
explicitly and pass it through the session data chain.
"""
from tea.global_vals import *
from tea.helpers.constants.default_values import DEFAULT_ALPHA_PARAMETER
from tea.runtimeDataStructures.dataset import Dataset
from tea.runtimeDataStructures.varData import VarData
from tea.runtimeDataStructures.combinedData import CombinedData
from tea.runtimeDataStructures.bivariateData import BivariateData
from tea.helpers.evaluateHelperMethods import get_data, compute_normal_distribution, compute_eq_variance
from tea.logging.tea_logger import TeaLogger

import attr
import z3
from typing import Dict, List


# ---------------------------------------------------------------------------
# Legacy module-level state. Only used by the backward-compatible shim
# functions at the bottom of this file. Explicit sessions never touch it.
# ---------------------------------------------------------------------------
alpha = DEFAULT_ALPHA_PARAMETER
MODE = 'strict'  # Default


@attr.s(hash=False, cmp=False, auto_attribs=True, init=False)
class StatVar:
    name: str

    def __init__(self, name):
        self.name = name
        self.__z3__ = z3.Bool(self.name)


class StatisticalTest:
    name: str
    variables: List[StatVar]
    test_properties: List["Property"]

    # properties_for_vars maps from a property to lists of indices
    # into test_vars that the property applies to. If the property
    # has arity 1, then it maps to a list of lists of length 1.
    # Similarly, if the property has arity 2, it maps to
    # a list of lists of length 2.
    properties_for_vars: Dict["Property", List[List[int]]]

    def __init__(self, context: "SolverContext", name, test_vars, test_properties=None, properties_for_vars=None):
        self.context = context
        self.name = name
        self.test_vars = test_vars

        if test_properties is None:
            test_properties = []

        if properties_for_vars is None:
            properties_for_vars = {}

        self.test_properties = test_properties

        self.properties_for_vars = {}
        for key in properties_for_vars:
            # properties_for_vars is a list of lists of variables. Get the indices of the variables
            # in each list.
            for prop_args in properties_for_vars[key]:
                indices = [self.test_vars.index(arg) for arg in prop_args]
                if key not in self.properties_for_vars.keys():
                    self.properties_for_vars[key] = []
                self.properties_for_vars[key].append(indices)

        # Create variable.
        self.__z3__ = z3.Bool(self.name)

        # Populate the owning context's tables.
        context.test_map[self.__z3__] = self
        context.all_tests_list.append(self)

    # Question: does this mean you can't consider two tests of the same type on different variables at once?
    def apply(self, *test_vars):
        assert len(test_vars) == len(self.test_vars)
        self.test_vars = test_vars

    def _populate_properties(self):
        # Populate all props
        self._properties = []
        for prop in self.test_properties:
            # This is weird, and doesn't really fit into property model
            # Alternative: Assign test-level properties to all the variables rather than
            # the test name
            variables = [v for v in self.test_vars]
            self._properties.append(prop(*variables))

        for prop in self.properties_for_vars:
            list_of_variable_indices = self.properties_for_vars[prop]
            for variable_indices in list_of_variable_indices:
                variable_indices = [self.test_vars[i] for i in variable_indices]
                self._properties.append(prop(*variable_indices))

    def properties(self):
        return self.test_properties + list(self.properties_for_vars)

    def query(self):
        context = self.context

        self._populate_properties()  # Apply to specific instance of variables
        conj = []
        for p in self._properties:  # combines Test and Var-specific properties
            # p.__z3__ = uninterpreted function
            # p.__var__ = instantiated z3 BoolVal
            # Add the uf, and interpret the uf as always ==ing the instantiated BoolVal
            conj += [p.__z3__]

            # Add property to property to function map
            context.property_to_function[p.__z3__] = p.property.function

        return conj

    def get_by_z3_var(self, var):
        """
        This relationship holds:
            t1 == StatisticalTet.get_by_z3_var(t1.__var__)
        """
        return self.context.test_map.get(var)


class Property:
    name: str
    description: str
    arity: int

    def __init__(self, context: "SolverContext", name, description, function=None, arity=1):
        self.context = context
        self.name = name
        self.description = description
        self.function = function
        self.arity = arity
        args = []
        for _ in range(self.arity):
            args.append(z3.BoolSort())
        args.append(z3.BoolSort())
        self.__z3__ = z3.Function(self.name, *args)  # e.g. continuous(x)
        self.__cache__ = {}

        # Populate the owning context's tables.
        context.property_map[self.__z3__] = self
        context.all_properties_list.append(self)

    def __str__(self):
        return f"property:{self.name}"

    def __repr__(self):
        return f"Property({self.name})"

    def __eq__(self, other):
        return self.arity == other.arity and self.name == other.name and self.function == other.function

    def __hash__(self):
        return hash((self.arity, self.name, self.function))

    def __call__(self, *var_names):
        if len(var_names) != self.arity:
            raise Exception(f"{self.name} property has arity {self.arity} " \
                            f"found {len(var_names)} arguments")
        cached = self.__cache__.get(tuple(var_names))
        if cached:
            return cached

        ap = AppliedProperty(self.context, self, var_names)
        # self.__cache__[tuple(var_names)] = ap
        return ap

    # Only used when the property must be changed depending on input data
    def _update(self, arity):
        self.arity = arity
        self.reset()

    def reset(self):
        args = []
        for _ in range(self.arity):
            args.append(z3.BoolSort())
        args.append(z3.BoolSort())
        self.__z3__ = z3.Function(self.name, *args)  # e.g. continuous(x)


class AppliedProperty:
    property: Property

    def __init__(self, context: "SolverContext", prop, pvars):
        self.context = context  # Bound to the analysis that applied this property
        self.property = prop  # e.g. continuous
        self.vars = pvars  # (StatVar(name='x'),)
        z3_args = []
        for tv in pvars:
            z3_args.append(tv.__z3__)
        self._name = self.property.name  # continuous
        self.__z3__ = prop.__z3__(*z3_args)  # e.g. continuous(x)

        # _name needs to be unique to avoid overwriting same property for different variables.
        # But if it is unique, it makes it more difficult to look up from the z3 model.
        # Solution: Make the name the same, but have it map to a list of variables that the
        # property applies to.
        if self._name not in context.property_var_map.keys():
            context.property_var_map[self._name] = []

        context.property_var_map[self._name].append(self.vars)

        self.property_test_results = None

    def __str__(self):
        return f"property_for_var:{self._name}"

    def __repr__(self):
        return f"AppliedProperty({self._name})"

    def __eq__(self, other):
        return self.property == other.property and self.vars == other.vars

    def get_by_z3_var(self, name):
        return self.context.property_var_map.get(name)


# ---------------------------------------------------------------------------
# Stateless property verification functions. They take everything they need
# as arguments, so they are safe to share across contexts.
# ---------------------------------------------------------------------------
def is_bivariate(dataset: Dataset, var_data: CombinedData, alpha):
    return len(var_data.vars) == 2


def is_multivariate(datset: Dataset, var_data: CombinedData, alpha):
    return len(var_data.vars) > 2


def has_one_x(dataset: Dataset, var_data: CombinedData, alpha):
    xs = var_data.get_explanatory_variables()

    return len(xs) == 1


def has_one_y(dataset: Dataset, var_data: CombinedData, alpha):
    ys = var_data.get_explained_variables()

    return len(ys) == 1


def has_paired_observations(dataset: Dataset, var_data: CombinedData, alpha):
    return var_data.properties[paired]


def has_independent_observations(dataset: Dataset, var_data: CombinedData, alpha):
    return not var_data.properties[paired]


def greater_than_5_frequency(dataset: Dataset, var_data: CombinedData, alpha):
    xs = var_data.get_explanatory_variables()
    ys = var_data.get_explained_variables()

    if len(xs) == 1:
        if len(ys) == 1:
            x = xs[0]
            y = ys[0]

            if x.is_categorical() and y.is_categorical():

                # Get the count for each category
                x_cat = [k for k , v in x.metadata[categories].items()]
                y_cat = [k for k , v in y.metadata[categories].items()]

                for xc in x_cat:
                    for yc in y_cat:
                        data = dataset.select(y.metadata[name], where=[f"{x.metadata[name]} == '{xc}'", f"{y.metadata[name]} == '{yc}'"])

                        # Check that the count is at least five for each of the (x,y) group pairs
                        if len(data) < 5:
                            return False

                return True
            else:
                return False
        else:
            raise ValueError(f"Currently, chi square requires/only supports 1 explained variable, instead received: {len(ys)} -- {ys}")
    else:
        x0 = xs[0]
        x1 = xs[1]

        if x0.is_categorical() and x1.is_categorical():
            # Get the count for each category
            x0_cat = [k for k , v in x0.metadata[categories].items()]
            x1_cat = [k for k , v in x1.metadata[categories].items()]

            for x0c in x0_cat:
                for x1c in x1_cat:
                    data = dataset.select(x1.metadata[name], where=[f"{x.metadata[name]} == '{xc}'", f"{x1.metadata[name]} == '{x1c}'"])

                    # Check that the count is at least five for each of the (x,x1) group pairs
                    if len(data) < 5:
                        return False
            return True
        else:
            return False


def has_equal_variance(dataset: Dataset, var_data: list, alpha):
    xs = []
    ys = []
    cat_xs = []
    cont_ys = []
    grouped_data = []


    if isinstance(var_data, CombinedData):
        xs = var_data.get_explanatory_variables()
        ys = var_data.get_explained_variables()

    else:
        for var in var_data:
            if var.role == iv_identifier or var.role == contributor_identifier:
                xs.append(var)
            if var.role == dv_identifier or var.role == outcome_identifier:
                ys.append(var)

    for x in xs:
        if x.is_categorical():
            cat_xs.append(x)

    for y in ys:
        if y.is_continuous():
            cont_ys.append(y)

    eq_var = (None, None)
    if cat_xs and cont_ys:
        for y in ys:
            for x in xs:
                cat = [k for k,v in x.metadata[categories].items()]
                for c in cat:
                    data = dataset.select(y.metadata[name], where=[f"{x.metadata[name]} == '{c}'"])
                    grouped_data.append(data)
                if isinstance(var_data, BivariateData):
                    # Equal variance
                    eq_var = compute_eq_variance(grouped_data)
                else:
                    eq_var = compute_eq_variance(grouped_data)

    if eq_var[0] is None and eq_var[1] is None:
        return False

    return eq_var[1] > alpha


def has_groups_normal_distribution(dataset, var_data, alpha):
    xs = []
    ys = []
    cat_xs = []
    cont_ys = []
    grouped_data = []
    result = None

    if isinstance(var_data, CombinedData):
        xs = var_data.get_explanatory_variables()
        ys = var_data.get_explained_variables()

    else:
        for var in var_data:
            if var.role == iv_identifier or var.role == contributor_identifier:
                xs.append(var)
            if var.role == dv_identifier or var.role == outcome_identifier:
                ys.append(var)

    for x in xs:
        if x.is_categorical():
            cat_xs.append(x)

    for y in ys:
        if y.is_continuous():
            cont_ys.append(y)

    if cat_xs and cont_ys:
        for y in ys:
            for x in xs:
                cat = [k for k,v in x.metadata[categories].items()]
                for c in cat:
                    if isinstance(c, str):
                        data = dataset.select(y.metadata[name], where=[f"{x.metadata[name]} == '{c}'"])
                    else:
                        data = dataset.select(y.metadata[name], where=[f"{x.metadata[name]} == {c}"])
                    grouped_data.append(data)

                for group in grouped_data:
                    result = compute_normal_distribution(group)
                    if result[1] <= alpha:
                        return False, result

    return True, result


def is_categorical_var(dataset, var_data, alpha):
    assert(len(var_data) == 1)
    assert(isinstance(var_data[0], VarData))

    return var_data[0].is_categorical()


def has_two_categories(dataset, var_data, alpha):
    assert(len(var_data) == 1)
    assert(isinstance(var_data[0], VarData))

    # First check that the variable is categorical
    assert(is_categorical_var(dataset, var_data, alpha))
    return len(var_data[0].metadata[categories].keys()) == 2


def has_two_or_more_categories(dataset, var_data, alpha):
    assert(len(var_data) == 1)
    assert(isinstance(var_data[0], VarData))

    # First check that the variable is categorical
    assert(is_categorical_var(dataset, var_data, alpha))
    return len(var_data[0].metadata[categories].keys()) >= 2


def has_three_or_more_categories(dataset, var_data, alpha):
    assert(len(var_data) == 1)
    assert(isinstance(var_data[0], VarData))

    # First check that the variable is categorical
    assert(is_categorical_var(dataset, var_data, alpha))
    return len(var_data[0].metadata[categories].keys()) >= 3


def is_continuous_var(dataset, var_data, alpha):
    assert(len(var_data) == 1)
    assert(isinstance(var_data[0], VarData))

    return var_data[0].is_continuous()


def is_continuous_or_ordinal_var(dataset, var_data, alpha):
    assert(len(var_data) == 1)
    assert(isinstance(var_data[0], VarData))

    return is_continuous_var(dataset, var_data, alpha) or var_data[0].is_ordinal()


def has_normal_distribution(dataset, var_data, alpha):
    assert(len(var_data) == 1)
    assert(isinstance(var_data[0], VarData))

    # Must be continuous to be normally distributed
    assert(is_continuous_var(dataset, var_data, alpha))
    # Get data from datasest using var_data's query
    data = get_data(dataset, var_data[0])
    norm_test_results = compute_normal_distribution(data)

    return (norm_test_results[1] > alpha), norm_test_results


class SolverContext:
    """Self-contained solver state for one analysis run.

    Holds the test/property registries, applied-property map, alpha and mode
    that previously lived in module globals. A context is never shared
    between analyses: discarding it after a run (including a failed one)
    leaves no properties behind to influence later test selection.
    """

    def __init__(self, mode: str = 'strict', alpha: float = DEFAULT_ALPHA_PARAMETER, session=None):
        self.mode = mode
        self.alpha = alpha
        # The session whose data chain this context executes against.
        self.session = session

        # Maps z3 variables representing tests back to the test objects.
        self.test_map = {}
        # All StatisticalTest instances constructed for this analysis.
        self.all_tests_list = []

        # Map from z3 variables representing applied properties back to them.
        self.property_var_map = {}
        # Map from property z3 functions back to the Property objects.
        self.property_map = {}
        # All Property instances in this context.
        self.all_properties_list = []
        # Map of property z3 terms to the functions that evaluate them.
        self.property_to_function = {}

        # Test instance handles (mirror the old module-level test variables).
        self.pearson_corr = None
        self.kendalltau_corr = None
        self.spearman_corr = None
        self.pointbiserial_corr_a = None
        self.pointbiserial_corr_b = None
        self.students_t = None
        self.paired_students_t = None
        self.welchs_t = None
        self.mannwhitney_u = None
        self.wilcoxon_signed_rank = None
        self.chi_square = None
        self.fishers_exact = None
        self.f_test = None
        self.rm_one_way_anova = None
        self.kruskall_wallis = None
        self.friedman = None
        self.factorial_ANOVA = None

        self._build_properties()

    def _build_properties(self):
        ctx = self

        # Test properties
        self.bivariate = Property(ctx, 'is_bivariate', "Exactly two variables involved in analysis", is_bivariate)
        self.multivariate = Property(ctx, 'is_multivariate', "More than two variables involved in analysis", is_multivariate)
        self.one_x_variable = Property(ctx, 'has_one_x', "Exactly one explanatory variable", has_one_x)
        self.one_y_variable = Property(ctx, 'has_one_y', "Exactly one explained variable", has_one_y)
        self.paired_obs = Property(ctx, 'has_paired_observations', "Paired observations", has_paired_observations)
        self.independent_obs = Property(ctx, 'has_independent_observations', "Independent (not paired) observations", has_independent_observations)
        self.greater_than_5_freq = Property(ctx, 'greater_than_5_freq', "Has a large sample size", greater_than_5_frequency, arity=2)

        self.test_props = [self.bivariate, self.one_x_variable, self.one_y_variable,
                           self.paired_obs, self.independent_obs]

        # Variable properties
        self.categorical = Property(ctx, 'is_categorical', "Variable is categorical", is_categorical_var)
        self.two_categories = Property(ctx, 'has_two_categories', "Variable has two categories", has_two_categories)
        self.two_or_more_categories = Property(ctx, 'has_two_or_more_categories', "Variable has two or more categories", has_two_or_more_categories)
        self.three_or_more_categories = Property(ctx, 'has_three_or_more_categories', "Variable has three or more categories", has_three_or_more_categories)
        self.continuous = Property(ctx, 'is_continuous', "Continuous (not categorical) data", is_continuous_var)
        self.continuous_or_ordinal = Property(ctx, 'is_continuous_or_ordinal', "Continuous OR ORDINAL (not nominal) data", is_continuous_or_ordinal_var)
        self.groups_normal = Property(ctx, 'is_groups_normal', "Groups are normally distributed", has_groups_normal_distribution, arity=2)
        self.normal = Property(ctx, 'is_normal', "Normal distribution", has_normal_distribution)
        self.eq_variance = Property(ctx, 'has_equal_variance', "Equal variance", has_equal_variance, arity=2)

    # Registry accessors (instance-level equivalents of the old module globals).
    def all_tests(self):
        return self.all_tests_list

    def all_props(self):
        return self.all_properties_list

    def set_mode(self, mode):
        self.mode = mode

    def reset_all_tests(self):
        self.test_map = {}
        self.all_tests_list = []
        self.property_to_function = {}
        self.property_var_map = {}

        self.pearson_corr = None
        self.kendalltau_corr = None
        self.spearman_corr = None
        self.pointbiserial_corr_a = None
        self.pointbiserial_corr_b = None
        self.students_t = None
        self.paired_students_t = None
        self.welchs_t = None
        self.mannwhitney_u = None
        self.wilcoxon_signed_rank = None
        self.chi_square = None
        self.fishers_exact = None
        self.f_test = None
        self.rm_one_way_anova = None
        self.kruskall_wallis = None
        self.friedman = None
        self.factorial_ANOVA = None

    def construct_test_axioms(self, solver):
        solver.add(self.paired_obs.__z3__ != self.independent_obs.__z3__)

    def construct_axioms(self, variables):  # List[StatVar]
        _axioms = []
        for var in variables:
            # A variable must be continuous or categorical, but not both.
            _axioms.append(z3.And(z3.Or(self.continuous(var).__z3__, self.categorical(var).__z3__),
                                  z3.Not(z3.And(self.continuous(var).__z3__, self.categorical(var).__z3__))))

            # If a variable is normal, then it cannot be categorical.
            _axioms.append(z3.Implies(self.normal(var).__z3__, z3.Not(self.categorical(var).__z3__)))

            # If a variable is continuous or ordinal, it must be continuous.
            _axioms.append(z3.Implies(self.continuous_or_ordinal(var).__z3__, self.continuous(var).__z3__))

        return _axioms

    # Constructs all the tests once per analysis.
    def construct_all_tests(self, combined_data: CombinedData):
        self.construct_bivariate_tests(combined_data)
        self.construct_mutlivariate_tests(combined_data)

    def construct_mutlivariate_tests(self, combined_data):
        self.construct_factorial_ANOVA(combined_data)

    def construct_factorial_ANOVA(self, combined_data: CombinedData):
        num_vars = len(combined_data.vars)

        x_vars = []
        y_vars = []
        all_vars = []
        for i in range(num_vars):
            name = 'x' + str(i)
            if (i <= num_vars - 2):  # All but the last var
                x = StatVar(name)
                x_vars.append(x)
                all_vars.append(x)
            else:
                assert(i == num_vars - 1)
                name = 'y' + str(i)
                y = StatVar(name)
                y_vars.append(y)  # last var
                all_vars.append(y)
        assert(num_vars == len(x_vars) + len(y_vars))

        list_x_vars = [[v] for v in x_vars]

        assert(len(y_vars) == 1)
        pairs_list = [[x, y_vars[0]] for x in x_vars]

        self.factorial_ANOVA = StatisticalTest(
            self, 'factorial_ANOVA', all_vars,  # Variable number of factors
            test_properties=[self.one_y_variable],
            properties_for_vars={
                self.continuous: [y_vars],
                self.categorical: list_x_vars,  # Variable number of factors
                self.two_or_more_categories: list_x_vars,
                self.groups_normal: pairs_list,  # Variable number of factors
                self.eq_variance: pairs_list  # Variable number of factors
            })

    def construct_bivariate_tests(self, combined_data: CombinedData):
        ctx = self

        # Bivariate analyses only make sense when there are two variables
        # in combined_data
        if len(combined_data.vars) == 2:

            # CORRELATIONS
            x0 = StatVar('x0')
            x1 = StatVar('x1')

            self.pearson_corr = StatisticalTest(ctx, 'pearson_corr', [x0, x1],
                                                test_properties=[ctx.bivariate],
                                                properties_for_vars={
                                                    ctx.continuous: [[x0], [x1]],
                                                    ctx.normal: [[x0], [x1]]
                                                })

            self.kendalltau_corr = StatisticalTest(ctx, 'kendalltau_corr', [x0, x1],
                                                   test_properties=[ctx.bivariate],
                                                   properties_for_vars={
                                                       ctx.continuous_or_ordinal: [[x0], [x1]]
                                                   })

            self.spearman_corr = StatisticalTest(ctx, 'spearman_corr', [x0, x1],
                                                 test_properties=[ctx.bivariate],
                                                 properties_for_vars={
                                                     ctx.continuous_or_ordinal: [[x0], [x1]]
                                                 })

            self.pointbiserial_corr_a = StatisticalTest(ctx, 'pointbiserial_corr_a', [x0, x1],
                                                        test_properties=[ctx.bivariate, ctx.independent_obs],
                                                        properties_for_vars={
                                                            ctx.continuous: [[x1]],
                                                            ctx.groups_normal: [[x0, x1]],
                                                            ctx.categorical: [[x0]],
                                                            ctx.two_categories: [[x0]],
                                                            ctx.eq_variance: [[x0, x1]]
                                                        })

            self.pointbiserial_corr_b = StatisticalTest(ctx, 'pointbiserial_corr_b', [x0, x1],
                                                        test_properties=[ctx.bivariate, ctx.independent_obs],
                                                        properties_for_vars={
                                                            ctx.continuous: [[x0]],
                                                            ctx.groups_normal: [[x1, x0]],
                                                            ctx.categorical: [[x1]],
                                                            ctx.two_categories: [[x1]],
                                                            ctx.eq_variance: [[x1, x0]]
                                                        })

            # T-TESTS
            x = StatVar('x')
            y = StatVar('y')

            self.students_t = StatisticalTest(ctx, 'students_t', [x, y],
                                              test_properties=[ctx.bivariate, ctx.one_x_variable,
                                                               ctx.one_y_variable, ctx.independent_obs],
                                              properties_for_vars={
                                                  ctx.categorical: [[x]],
                                                  ctx.two_categories: [[x]],
                                                  ctx.continuous: [[y]],
                                                  ctx.eq_variance: [[x, y]],
                                                  ctx.groups_normal: [[x, y]]
                                              })

            self.welchs_t = StatisticalTest(ctx, 'welchs_t', [x, y],
                                            test_properties=[ctx.bivariate, ctx.one_x_variable,
                                                             ctx.one_y_variable, ctx.independent_obs],
                                            properties_for_vars={
                                                ctx.categorical: [[x]],
                                                ctx.two_categories: [[x]],
                                                ctx.continuous: [[y]],
                                                ctx.groups_normal: [[x, y]]
                                            })

            self.mannwhitney_u = StatisticalTest(ctx, 'mannwhitney_u', [x, y],
                                                 test_properties=[ctx.one_x_variable, ctx.one_y_variable,
                                                                  ctx.independent_obs],
                                                 properties_for_vars={
                                                     ctx.categorical: [[x]],
                                                     ctx.two_categories: [[x]],
                                                     ctx.continuous_or_ordinal: [[y]],
                                                 })

            self.paired_students_t = StatisticalTest(ctx, 'paired_students_t', [x, y],
                                                     test_properties=[ctx.bivariate, ctx.one_x_variable,
                                                                      ctx.one_y_variable, ctx.paired_obs],
                                                     properties_for_vars={
                                                         ctx.categorical: [[x]],
                                                         ctx.two_categories: [[x]],
                                                         ctx.continuous: [[y]],
                                                         ctx.groups_normal: [[x, y]],
                                                         ctx.eq_variance: [[x, y]]
                                                     })

            self.wilcoxon_signed_rank = StatisticalTest(ctx, 'wilcoxon_signed_rank', [x, y],
                                                        test_properties=[ctx.bivariate, ctx.one_x_variable,
                                                                         ctx.one_y_variable, ctx.paired_obs],
                                                        properties_for_vars={
                                                            ctx.categorical: [[x]],
                                                            ctx.two_categories: [[x]],
                                                            ctx.continuous: [[y]],
                                                        })

            # CONTINGENCY TABLES (Categorical data)
            self.chi_square = StatisticalTest(ctx, 'chi_square', [x, y],
                                              test_properties=[ctx.bivariate, ctx.independent_obs,
                                                               ctx.greater_than_5_freq],
                                              properties_for_vars={
                                                  ctx.categorical: [[x], [y]],
                                                  ctx.two_or_more_categories: [[x], [y]]
                                              })

            self.fishers_exact = StatisticalTest(ctx, 'fishers_exact', [x, y],
                                                 test_properties=[ctx.bivariate, ctx.independent_obs],
                                                 properties_for_vars={
                                                     ctx.categorical: [[x], [y]],
                                                     ctx.two_categories: [[x], [y]]
                                                 })

            # ANOVAs
            self.f_test = StatisticalTest(ctx, 'f_test', [x, y],  # Variable number of factors
                                          test_properties=[ctx.independent_obs, ctx.one_x_variable,
                                                           ctx.one_y_variable],
                                          properties_for_vars={
                                              ctx.continuous: [[y]],
                                              ctx.categorical: [[x]],  # Variable number of factors
                                              ctx.two_or_more_categories: [[x]],
                                              ctx.groups_normal: [[x, y]],  # Variable number of factors
                                              ctx.eq_variance: [[x, y]]  # Variable number of factors
                                          })

            self.kruskall_wallis = StatisticalTest(ctx, 'kruskall_wallis', [x, y],
                                                   test_properties=[ctx.independent_obs, ctx.one_x_variable,
                                                                    ctx.one_y_variable],
                                                   properties_for_vars={
                                                       ctx.continuous: [[y]],
                                                       ctx.categorical: [[x]],  # Variable number of factors
                                                       ctx.two_or_more_categories: [[x]]
                                                   })

            self.rm_one_way_anova = StatisticalTest(ctx, 'rm_one_way_anova', [x, y],
                                                     test_properties=[ctx.paired_obs, ctx.one_x_variable,
                                                                      ctx.one_y_variable],
                                                     properties_for_vars={
                                                         ctx.continuous: [[y]],
                                                         ctx.categorical: [[x]],  # Variable number of factors
                                                         ctx.two_or_more_categories: [[x]],
                                                         ctx.groups_normal: [[x, y]],  # Variable number of factors
                                                         ctx.eq_variance: [[x, y]]  # Variable number of factors
                                                     })

            self.friedman = StatisticalTest(ctx, 'friedman', [x, y],
                                            test_properties=[ctx.paired_obs, ctx.one_x_variable,
                                                             ctx.one_y_variable],
                                            properties_for_vars={
                                                ctx.continuous: [[y]],
                                                ctx.categorical: [[x]],  # Variable number of factors
                                                ctx.three_or_more_categories: [[x]],
                                            })

    # Verify the property against data held on this context's analysis chain.
    def verify_prop(self, dataset: Dataset, combined_data: CombinedData, prop: AppliedProperty):
        if len(prop.vars) == len(combined_data.vars):
            kwargs = {'dataset': dataset, 'var_data': combined_data, 'alpha': self.alpha}
            if self.property_to_function == {}:
                prop_val = prop.property.function(**kwargs)
            else:
                prop_val = self.property_to_function[prop.__z3__](**kwargs)
        else:
            assert (len(prop.vars) < len(combined_data.vars))
            var_data = []
            # For each of the variables for which we are checking the current prop
            for test_var in prop.vars:
                # For each variable in all variables in combined data
                for var in combined_data.vars:
                    if var.metadata[name] == test_var.name:
                        var_data.append(var)
            kwargs = {'dataset': dataset, 'var_data': var_data, 'alpha': self.alpha}
            if self.property_to_function == {}:
                prop_val = prop.property.function(**kwargs)
            else:
                prop_val = self.property_to_function[prop.__z3__](**kwargs)

        ret_val = None
        if isinstance(prop_val, tuple):
            assert len(prop_val) == 2
            ret_val = prop_val[0]
            prop.property_test_results = prop_val[1]
        else:
            ret_val = prop_val

        return ret_val

    # Assumes properties to hold
    def assume_properties(self, stat_var_map, assumptions: Dict[str, str], solver, dataset, combined_data):
        ctx = self
        tea_logger = TeaLogger.get_logger()

        assumed_props = []

        # Go through all assumptions
        for a in assumptions:
            if a in alpha_keywords:
                ctx.alpha = float(assumptions[a])
            elif a in assumptions_to_properties:
                # Apply corresponding properties to all variables for which the property is assumed
                for prop in ctx.all_props():
                    if prop.name in assumptions_to_properties[a]:
                        # Convert assumptions to use correct statistical variable names.
                        for var in assumptions[a]:
                            # This is really hacky -- should probably change the
                            # incoming assumptions data structure before calling this method
                            if isinstance(var, str):
                                assert var in stat_var_map
                                ap = prop(stat_var_map[var])
                                assumed_props.append(ap)

                                # CHECK ASSUMPTIONS HERE
                                val = ctx.verify_prop(dataset, combined_data, ap)
                                if ctx.mode == 'strict':
                                    tea_logger.log_debug(f"Running under STRICT mode.")
                                    if val:
                                        tea_logger.log_debug(f"User asserted property: {prop.name} is supported by statistical checking. Tea agrees with the user.")
                                    else:
                                        tea_logger.log_debug(f"User asserted property: {prop.name}, but is NOT supported by statistical checking. Tea will override user assertion.")
                                    solver.add(ap.__z3__ == z3.BoolVal(val))
                                elif ctx.mode == 'relaxed':
                                    tea_logger.log_debug(f"Running under RELAXED mode.")
                                    if val:
                                        tea_logger.log_debug(f"User asserted property: {prop.name} is supported by statistical checking. Tea agrees with the user.")
                                    else:
                                        tea_logger.log_debug(f"User asserted property: {prop.name}, but is NOT supported by statistical checking. User assertion will be considered true.")

                                    solver.add(ap.__z3__ == z3.BoolVal(True))
                                else:
                                    raise ValueError(f"Invalid MODE: {ctx.mode}")
                            else:
                                assert isinstance(var, list)
                                # Checks that only the assumptions pertaining to the hypothesis are checked and added
                                if all(v in stat_var_map for v in var):
                                    stat_vars = [stat_var_map[v] for v in var]
                                    ap = prop(*stat_vars)
                                    assumed_props.append(ap)

                                    # CHECK ASSUMPTIONS HERE
                                    val = ctx.verify_prop(dataset, combined_data, ap)
                                    if ctx.mode == 'strict':
                                        tea_logger.log_debug(f"Running under STRICT mode.")
                                        if val:
                                            tea_logger.log_debug(f"User asserted property: {prop.name} is supported by statistical checking. Tea agrees with the user.")
                                        else:
                                            tea_logger.log_debug(f"User asserted property: {prop.name}, but is NOT supported by statistical checking. Tea will override user assertion.")
                                        solver.add(ap.__z3__ == z3.BoolVal(val))
                                    elif ctx.mode == 'relaxed':
                                        tea_logger.log_debug(f"Running under RELAXED mode.")
                                        if val:
                                            tea_logger.log_debug(f"User asserted property: {prop.name} is supported by statistical checking. Tea agrees with the user.")
                                        else:
                                            tea_logger.log_debug(f"User asserted property: {prop.name}, but is NOT supported by statistical checking. User assertion will be considered true.")
                                        solver.add(ap.__z3__ == z3.BoolVal(True))  # override user
                                    else:
                                        raise ValueError(f"Invalid MODE: {ctx.mode}")
            else:
                pass
        return assumed_props

    # Helper for synthesize tests
    def is_assumed_prop(self, assumed_props, prop):
        for ap in assumed_props:
            if prop == ap:
                return True

        return False

    # Problem statement: Given a set of properties, tell me which tests are valid to run
    # This is a concrete (rather than symbolic) problem
    # @param combined_data CombinedData object
    # @returns list of Property objects that combined_data exhibits
    def synthesize_tests(self, dataset: Dataset, assumptions: Dict[str, str], combined_data: CombinedData):
        ctx = self
        ctx.reset_all_tests()

        tea_logger = TeaLogger.get_logger()
        ctx.construct_all_tests(combined_data)

        stat_var_map = {}

        # Reorder variables so that y var is at the end
        combined_data._update_vars()

        # Compute unique statisical variable names from the combined data.
        combined_data_vars = []
        for v in combined_data.vars:
            var = StatVar(v.metadata[name])
            stat_var_map[v.metadata[name]] = var
            combined_data_vars.append(var)

        # Assume properties based on user assumptions and mode
        solver = z3.Solver()
        assumed_props = ctx.assume_properties(stat_var_map, assumptions, solver, dataset, combined_data)

        # Update the arity of test-level properties
        for prop in ctx.test_props:
            prop._update(len(combined_data.vars))

        # Apply all tests to the variables we are considering now in combined_data
        for test in ctx.all_tests():
            test.apply(*combined_data_vars)

        solver.push()  # Create backtracking point
        model = None  # Store model

        # For each test, add it to the solver as a constraint.
        # Add the tests and their properties
        for test in ctx.all_tests():
            tea_logger.log_debug(f"\nCurrently considering {test.name}")
            solver.add(test.__z3__ == z3.And(*test.query()))
            solver.add(test.__z3__ == z3.BoolVal(True))

            # Check the model
            result = solver.check()
            if result == z3.unsat:
                tea_logger.log_debug("Test is unsat.\n")
                solver.pop()
            elif result == z3.unknown:
                print("failed to solve")
                try:
                    pass
                except z3.Z3Exception:
                    return
            else:
                model = solver.model()
                test_invalid = False
                val = False
                # Does the test apply?
                if model and z3.is_true(model.evaluate(test.__z3__)):
                    # Verify the properties for that test
                    for prop in test._properties:
                        if ctx.is_assumed_prop(assumed_props, prop):
                            tea_logger.log_debug(f"User asserted property: {prop._name}.")
                            val = True
                            solver.add(prop.__z3__ == z3.BoolVal(val))
                        else:
                            tea_logger.log_debug(f"Testing assumption: {prop._name}.")

                            # Does this property need to hold for the test to be valid?
                            # Is there data to evaluate properties?
                            if not dataset.data.empty:
                                # If so, verify that the property does hold
                                if model and z3.is_true(model.evaluate(prop.__z3__)):
                                    val = ctx.verify_prop(dataset, combined_data, prop)
                                    if val:
                                        tea_logger.log_debug(f"Property holds.")
                                    else:  # The property does not verify
                                        assert (val == False)
                                        tea_logger.log_debug(f"Property FAILS")
                                        solver.pop()  # remove the last test
                                        test_invalid = True
                                        model = None
                                    solver.add(prop.__z3__ == z3.BoolVal(val))
            solver.push()  # Push latest state as backtracking point

        solver.check()
        model = solver.model()  # final model
        tests_to_conduct = []
        for test in ctx.all_tests():
            if model and z3.is_true(model.evaluate(test.__z3__)):
                tests_to_conduct.append(test.name)
            elif not model:  # No test applies
                pass

        return tests_to_conduct

    def which_props(self, tests_names: list, var_names: List[str]):
        ctx = self
        stat_vars: List[StatVar] = []
        for var_name in var_names:
            stat_vars.append(StatVar(var_name))

        axioms = ctx.construct_axioms(stat_vars)

        # Update the arity of test-level properties to match the number of
        # variables under analysis (mirrors synthesize_tests); without this,
        # properties such as has_one_x keep arity 1 while applied to a pair.
        for prop in ctx.test_props:
            prop._update(len(stat_vars))

        tests: List[StatisticalTest] = []
        for test_name in tests_names:
            for test in ctx.all_tests():
                if test.name == test_name:
                    test.apply(*stat_vars)
                    test._populate_properties()
                    tests.append(test)

        test_queries = []
        important_soft_constraints = []
        for test in tests:
            # The Bool representing whether the test can be run with the given property
            # values is a hard constraint because we want to satisfy all tests, but may
            # not be able to.
            important_soft_constraints.append(test.__z3__)

            # The relationship of the properties of variables required to run a specific
            # test is an axiom and cannot be violated.
            axioms.append(test.__z3__ == z3.And(*test.query()))

            # test_queries contains the individual property constraints (e.g. continuous(x)).
            # These will all be soft constraints with low weight because they may need to be
            # violated.
            test_queries += test.query()

        s = z3.Optimize()

        # Set weight of tests to large number so they are satisfied over properties.
        for constraint in important_soft_constraints:
            s.add_soft(constraint, 1000)

        # Prefer to violate properties by giving them low weight.
        for soft_constraint in test_queries:
            s.add_soft(soft_constraint, 1)

        # Axioms cannot be violated.
        for axiom in axioms:
            s.add(axiom)

        result = s.check()
        if result == z3.unsat:
            print("no solution")
        elif result == z3.unknown:
            print("failed to solve")
            try:
                print(f"Failed to solve: {s.model()}")
            except z3.Z3Exception:
                return
        else:
            model = s.model()
            _tests_and_properties = {}
            _test_to_broken_properties = {}
            for test in tests:
                test_name = test.name
                _tests_and_properties[test_name] = {}
                _test_to_broken_properties[test_name] = []

                # TODO: Unprotect test._properties.
                for test_property in test._properties:
                    property_identifier = ""  # test_property._name
                    for test_var in test_property.vars:
                        property_identifier += "variable %s : " % test_var.name
                    property_identifier += test_property._name
                    property_result = bool(model.evaluate(test_property.__z3__))
                    _tests_and_properties[test_name][property_identifier] = property_result
                    if not property_result:
                        _test_to_broken_properties[test_name].append(property_identifier)

            return _tests_and_properties, _test_to_broken_properties


# ---------------------------------------------------------------------------
# Backward-compatible shims. These delegate to a legacy SolverContext so that
# code importing the old module-level functions keeps working. Explicit
# sessions create and pass their own contexts instead.
# ---------------------------------------------------------------------------
_LEGACY = SolverContext(mode=MODE, alpha=alpha)


def set_mode(mode):
    global MODE
    MODE = mode
    _LEGACY.set_mode(mode)


def all_tests():
    """A helper for accessing the legacy context's set of tests."""
    return _LEGACY.all_tests()


def all_props():
    """A helper for accessing the legacy context's set of properties."""
    return _LEGACY.all_props()


def reset_all_tests():
    _LEGACY.reset_all_tests()


def synthesize_tests(dataset: Dataset, assumptions: Dict[str, str], combined_data: CombinedData):
    return _LEGACY.synthesize_tests(dataset, assumptions, combined_data)


def which_props(tests_names: list, var_names: List[str]):
    return _LEGACY.which_props(tests_names, var_names)
