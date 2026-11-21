from tea.api import (
                    data,
                    define_variables,
                    define_study_design,
                    assume,
                    hypothesize,
                    download_data,
                    divine_properties,
                    attach_covariates,
                    get_covariate_evidence,
                )
from tea.hypotheses import compare, relationship
from tea.runtimeDataStructures.covariateJoin import (
                    CovariateJoinSpec,
                    CovariateJoinError,
                    OnOverlap,
                    OnDuplicate,
                    OnMissing,
                )
