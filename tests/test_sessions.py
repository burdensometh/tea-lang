"""Tests for explicit, isolated analysis sessions.

These tests pin the session guarantees requested for Tea:

* each :class:`tea.api.Session` owns its data snapshot, variables, study
  design, assumptions, alpha and solver state;
* two sessions can be driven interleaved in one process without interference;
* VarData/applied properties/ResultData stay bound to the owning session;
* changing data or definitions after an analysis cannot rewrite its results;
* a failed/finished analysis leaves no properties behind for the next run,
  because every run gets a fresh, discarded :class:`SolverContext`.
"""
import os

import pandas as pd
import pytest

import tea
from tea.api import Session, create_session
from tea.z3_solver.solver import SolverContext

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _path(filename):
    return os.path.join(DATA_DIR, filename)


def _cats_session(alpha=0.05, mode=None):
    s = create_session("cats")
    s.data(_path("catsData.csv"))
    s.define_variables([
        {"name": "Training", "data type": "nominal",
         "categories": ["Food as Reward", "Affection as Reward"]},
        {"name": "Dance", "data type": "nominal", "categories": ["Yes", "No"]},
    ])
    s.define_study_design({
        "study type": "observational study",
        "contributor variables": "Training",
        "outcome variables": "Dance",
    })
    s.assume({"Type I (False Positive) Error Rate": alpha}, mode=mode)
    return s


def _uscrime_session(alpha=0.05):
    s = create_session("uscrime")
    s.data(_path("UScrime.csv"))
    s.define_variables([
        {"name": "So", "data type": "nominal", "categories": [0, 1]},
        {"name": "Prob", "data type": "ratio", "range": [0, 1]},
    ])
    s.define_study_design({
        "study type": "observational study",
        "contributor variables": "So",
        "outcome variables": "Prob",
    })
    s.assume({"Type I (False Positive) Error Rate": alpha})
    return s


class TestSessionConstruction:
    def test_create_session_returns_distinct_instances(self):
        s1 = create_session()
        s2 = create_session()
        assert isinstance(s1, Session)
        assert s1 is not s2

    def test_new_session_has_independent_state(self):
        s1 = _cats_session()
        s2 = create_session()
        # Configuring s1 must not populate the untouched session s2.
        assert s2.vars_objs == []
        assert s2.study_design is None
        assert s2.assumptions == {}
        assert s1.vars_objs != []

    def test_solver_contexts_are_independent(self):
        c1 = SolverContext()
        c2 = SolverContext()
        assert c1.all_tests_list is not c2.all_tests_list
        assert c1.all_properties_list is not c2.all_properties_list
        assert c1.test_map is not c2.test_map
        assert c1.property_var_map is not c2.property_var_map
        # Property/test descriptors are bound to their owning context.
        assert c1.categorical is not c2.categorical
        assert c1.categorical.context is c1
        assert c2.categorical.context is c2


class TestInterleavedAnalyses:
    def test_interleaved_sessions_do_not_share_results_or_context(self):
        s_a = _uscrime_session()
        s_b = _cats_session(alpha=0.01)

        r_a1 = s_a.hypothesize(["So", "Prob"], ["So:1 > 0"])
        r_b = s_b.hypothesize(["Training", "Dance"])
        r_a2 = s_a.hypothesize(["So", "Prob"], ["So:1 > 0"])

        # Results are bound to their owning session.
        assert r_a1.session is s_a
        assert r_a2.session is s_a
        assert r_b.session is s_b
        # Every run gets its own context, even within one session.
        assert r_a1.context is not r_a2.context
        assert r_a1.context is not r_b.context
        assert r_b.context.session is s_b

    def test_alpha_is_isolated_between_sessions(self):
        s_a = _uscrime_session(alpha=0.05)
        s_b = _cats_session(alpha=0.01)

        r_a = s_a.hypothesize(["So", "Prob"], ["So:1 > 0"])
        r_b = s_b.hypothesize(["Training", "Dance"])

        alpha_a = list(r_a.test_to_results.values())[0].alpha
        alpha_b = list(r_b.test_to_results.values())[0].alpha
        assert alpha_a == 0.05
        assert alpha_b == 0.01

    def test_test_selection_follows_each_session_data_chain(self):
        s_a = _uscrime_session()
        s_b = _cats_session()

        r_a = s_a.hypothesize(["So", "Prob"], ["So:1 > 0"])
        r_b = s_b.hypothesize(["Training", "Dance"])

        tests_a = set(r_a.test_to_results.keys())
        tests_b = set(r_b.test_to_results.keys())
        assert "chi_square" not in tests_a
        assert "fishers_exact" not in tests_a
        assert tests_b <= {"chi_square", "fishers_exact"}
        # Interpretation reads the per-result context registry, not a global one.
        assert set(r_a.test_to_assumptions.keys()) == tests_a
        assert set(r_b.test_to_assumptions.keys()) == tests_b

    def test_mode_does_not_leak_between_sessions(self):
        s_relaxed = _cats_session(mode="relaxed")
        s_strict = _cats_session()
        assert s_relaxed.MODE == "relaxed"
        assert s_strict.MODE == "strict"
        # Running the relaxed session does not change the strict one.
        s_relaxed.hypothesize(["Training", "Dance"])
        assert s_strict.MODE == "strict"


class TestResultImmutability:
    def test_redefining_data_cannot_rewrite_prior_result(self):
        s = _cats_session()
        before = s.hypothesize(["Training", "Dance"])
        before_p = {k: v.p_value for k, v in before.test_to_results.items()}
        before_tests = set(before.test_to_results.keys())

        # Repoint the session at a different dataset and definitions.
        s.data(_path("UScrime.csv"))
        s.define_variables([
            {"name": "So", "data type": "nominal", "categories": [0, 1]},
            {"name": "Prob", "data type": "ratio", "range": [0, 1]},
        ])
        s.define_study_design({
            "study type": "observational study",
            "contributor variables": "So",
            "outcome variables": "Prob",
        })
        s.assume({"Type I (False Positive) Error Rate": 0.01})

        assert {k: v.p_value for k, v in before.test_to_results.items()} == before_p
        assert set(before.test_to_results.keys()) == before_tests
        # The stored result still remembers the session it came from.
        assert before.session is s

    def test_dataframe_mutation_does_not_affect_snapshot(self):
        df = pd.read_csv(_path("catsData.csv"))
        s = create_session()
        s.data(df)
        s.define_variables([
            {"name": "Training", "data type": "nominal",
             "categories": ["Food as Reward", "Affection as Reward"]},
            {"name": "Dance", "data type": "nominal", "categories": ["Yes", "No"]},
        ])
        s.define_study_design({
            "study type": "observational study",
            "contributor variables": "Training",
            "outcome variables": "Dance",
        })
        s.assume({"Type I (False Positive) Error Rate": 0.05})

        result = s.hypothesize(["Training", "Dance"])
        snapshot_rows = len(s.dataset_obj.data)
        # Destroy the caller's frame; the session snapshot must be unaffected.
        df.drop(df.index, inplace=True)
        assert len(s.dataset_obj.data) == snapshot_rows
        assert result.session is s


class TestNoResidualState:
    def test_each_run_starts_with_fresh_context_registries(self):
        s = _cats_session()
        r1 = s.hypothesize(["Training", "Dance"])
        r2 = s.hypothesize(["Training", "Dance"])

        assert r1.context is not r2.context
        # The second context's applied properties reference only its own vars.
        for applied in r2.context.property_var_map.values():
            for var_tuple in applied:
                assert all(v.name in ("Training", "Dance") for v in var_tuple)
        # Registry containers are distinct objects across runs.
        assert r1.context.test_map is not r2.context.test_map
        assert r1.context.all_properties_list is not r2.context.all_properties_list

    def test_discarded_context_cannot_influence_another_session(self):
        # Simulate residue left inside a failed run's context: it is local to
        # that context and invisible to any other context.
        c_failed = SolverContext()
        c_healthy = SolverContext()
        c_failed.all_tests_list.append("residue-from-failure")
        assert "residue-from-failure" not in c_healthy.all_tests_list
        assert c_healthy.all_tests() == []

    def test_explicit_session_does_not_touch_default_globals(self):
        import tea.api as api

        default_results_before = dict(api.all_results)
        default_mode = api.MODE

        s = _cats_session(mode="strict")
        s.hypothesize(["Training", "Dance"])

        # The module-level default-session mirrors are untouched by the
        # explicit session's run.
        assert api.all_results == default_results_before
        assert api.MODE == default_mode


class TestModuleLevelApiBackwardCompatibility:
    def test_module_functions_delegate_to_default_session(self, monkeypatch):
        # Reload to reset the default session state.
        import importlib
        import tea.api as api
        importlib.reload(api)

        assert api.vars_objs == []
        api.define_variables([
            {"name": "Dance", "data type": "nominal", "categories": ["Yes", "No"]},
        ])
        # Legacy module globals mirror the default session after each call.
        assert [v.name for v in api.vars_objs] == ["Dance"]
        assert api.vars_objs is api._default_session.vars_objs

        api.data(_path("catsData.csv"))
        assert api.dataset_path == api._default_session.dataset_path
        api.assume({"Type I (False Positive) Error Rate": 0.01})
        assert api.alpha == 0.01
        assert api.assumptions["alpha"] == 0.01

    def test_tea_package_exports_session_api(self):
        assert hasattr(tea, "Session")
        assert hasattr(tea, "create_session")
        assert callable(tea.create_session)


class TestDivineProperties:
    def test_divine_properties_runs_on_session_context(self, capsys):
        # Regression: the legacy module-level divine_properties was broken by
        # missing imports; the session version runs through SolverContext.
        s = _uscrime_session()
        s.divine_properties(["So", "Prob"], ["mannwhitney_u", "students_t"])
        out = capsys.readouterr().out
        assert "Properties:" in out
        assert "mannwhitney_u" in out and "students_t" in out

    def test_divine_properties_does_not_pollute_another_session(self):
        s1 = _uscrime_session()
        s2 = _cats_session()
        s1.divine_properties(["So", "Prob"], ["mannwhitney_u", "students_t"])
        # s2 remains fully usable and selects its own (categorical) tests.
        r2 = s2.hypothesize(["Training", "Dance"])
        assert set(r2.test_to_results.keys()) <= {"chi_square", "fishers_exact"}
