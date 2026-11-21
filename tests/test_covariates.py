"""Tests for external participant-scoped covariate joins.

Covers the temporal join contract:

* event/effective/optional-expiry declaration,
* exactly one in-force record per event,
* no backfill from future records,
* explicit policies for overlapping versions, duplicate keys and missing matches,
* event-row identity and join evidence,
* one snapshot shared by the variable schema, attribute validation and
  hypothesize, with no sample-size inflation,
* caller inputs untouched and time precision/time zone agreement enforced.
"""

import os

import numpy as np
import pandas as pd
import pytest

import tea
from tea.ast import DataType
from tea.build import nominal, ratio
from tea.runtimeDataStructures.covariateJoin import (
    CovariateJoinError,
    CovariateJoinSpec,
    OnDuplicate,
    OnMissing,
    OnOverlap,
    as_of_join,
    make_snapshot,
    validate_attribute_schema,
)
from tea.runtimeDataStructures.dataset import Dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_events():
    return pd.DataFrame(
        {
            "eid": [1, 2, 3, 4, 5, 6],
            "pid": ["a", "a", "b", "b", "c", "a"],
            "t": pd.to_datetime(
                [
                    "2020-01-15",
                    "2020-03-01",
                    "2020-01-15",
                    "2020-02-15",
                    "2020-02-01",
                    "2020-04-01",
                ]
            ),
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def make_history():
    return pd.DataFrame(
        {
            "pid": ["a", "a", "b"],
            "eff": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-10"]),
            "exp": pd.to_datetime(["2020-02-01", "2020-03-15", "2020-02-01"]),
            "grp": ["x", "y", "x"],
            "age": [30.0, 31.0, 50.0],
        }
    )


def spec(**overrides):
    base = dict(
        event_time="t",
        effective_time="eff",
        expires_time="exp",
        participant_key="pid",
        on_missing="allow_na",
    )
    base.update(overrides)
    return CovariateJoinSpec(**base)


def covariate_variables():
    return [nominal("grp", ["x", "y"]), ratio("age")]


# ---------------------------------------------------------------------------
# As-of matching
# ---------------------------------------------------------------------------


class TestAsOfSelection:
    def test_uses_the_record_in_force_at_event_time(self):
        out, evidence = as_of_join(make_events(), make_history(), spec())

        by_event = out.set_index("eid")
        assert by_event.loc[1, "grp"] == "x"  # a, effective 01-01
        assert by_event.loc[2, "grp"] == "y"  # a, effective 02-01
        assert by_event.loc[3, "grp"] == "x"  # b, effective 01-10
        # eid 4 (b) is past its expiry, 5 (c) has no history, 6 (a) is past
        # the last version's expiry.
        assert evidence.n_matched == 3
        assert evidence.n_missing == 3

    def test_effective_time_is_inclusive(self):
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-02-01"])})
        out, _ = as_of_join(events, make_history(), spec())
        assert out.loc[0, "grp"] == "y"

    def test_expiry_time_is_exclusive(self):
        events = pd.DataFrame(
            {"pid": ["a", "a"], "t": pd.to_datetime(["2020-02-29", "2020-03-15"])}
        )
        out, _ = as_of_join(events, make_history(), spec())
        assert out.loc[0, "grp"] == "y"
        assert out.loc[1, "grp"] is np.nan or pd.isna(out.loc[1, "grp"])

    def test_future_records_do_not_backfill_the_past(self):
        events = pd.DataFrame(
            {"pid": ["a", "a"], "t": pd.to_datetime(["2020-01-01", "2020-06-01"])}
        )
        history = pd.DataFrame(
            {"pid": ["a"], "eff": pd.to_datetime(["2020-03-01"]), "v": [7]}
        )
        out, evidence = as_of_join(
            events, history, spec(expires_time=None, on_missing="allow_na")
        )
        # The earlier event predates the only record, so it stays missing.
        assert pd.isna(out.loc[0, "v"])
        assert out.loc[1, "v"] == 7
        assert evidence.n_missing == 1

    def test_gap_between_expiry_and_next_version_is_missing(self):
        events = pd.DataFrame(
            {"pid": ["a", "a"], "t": pd.to_datetime(["2020-03-01", "2020-04-15"])}
        )
        history = pd.DataFrame(
            {
                "pid": ["a", "a"],
                "eff": pd.to_datetime(["2020-01-01", "2020-04-01"]),
                "exp": pd.to_datetime(["2020-02-01", "2020-05-01"]),
                "v": [1, 2],
            }
        )
        out, _ = as_of_join(events, history, spec())
        assert pd.isna(out.loc[0, "v"])
        assert out.loc[1, "v"] == 2

    def test_exactly_one_record_matched_per_event(self):
        out, evidence = as_of_join(make_events(), make_history(), spec())
        assert len(out) == len(make_events())
        assert out["__tea_join_matched_row"].notna().sum() == evidence.n_matched
        # matched_row is a single history-row id per event, never a list.
        assert all(
            pd.isna(value) or isinstance(value, (int, np.integer))
            for value in out["__tea_join_matched_row"]
        )


# ---------------------------------------------------------------------------
# Overlap policy
# ---------------------------------------------------------------------------


def overlap_frames():
    events = pd.DataFrame(
        {
            "pid": ["a", "a", "a"],
            "t": pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15"]),
        }
    )
    history = pd.DataFrame(
        {
            "pid": ["a", "a"],
            "eff": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "exp": pd.to_datetime(["2020-03-01", "2020-04-01"]),
            "v": [1, 2],
        }
    )
    return events, history


class TestOverlapPolicy:
    def test_error_is_the_default_and_names_the_resolution(self):
        events, history = overlap_frames()
        with pytest.raises(CovariateJoinError, match="Overlapping versions"):
            as_of_join(events, history, CovariateJoinSpec(
                event_time="t", effective_time="eff", expires_time="exp",
                participant_key="pid",
            ))

    def test_latest_effective_truncates_the_earlier_version(self):
        events, history = overlap_frames()
        out, evidence = as_of_join(
            events, history, spec(on_overlap="latest_effective")
        )
        assert out["v"].tolist() == [1, 2, 2]
        assert evidence.n_overlaps_resolved == 1

    def test_earliest_effective_defers_the_later_version(self):
        events, history = overlap_frames()
        out, evidence = as_of_join(
            events, history, spec(on_overlap="earliest_effective")
        )
        assert out["v"].tolist() == [1, 1, 2]
        assert evidence.n_overlaps_resolved == 1

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(CovariateJoinError, match="Unknown on_overlap"):
            CovariateJoinSpec(
                event_time="t", effective_time="eff", participant_key="pid",
                on_overlap="yolo",
            )


# ---------------------------------------------------------------------------
# Duplicate policy
# ---------------------------------------------------------------------------


class TestDuplicatePolicy:
    def _frames(self, conflicting):
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-01-15"])})
        history = pd.DataFrame(
            {
                "pid": ["a", "a", "a"],
                "eff": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"]),
                "exp": pd.to_datetime(["2020-02-01", "2020-02-01", "2020-03-01"]),
                "v": [1, 2 if conflicting else 1, 3],
            }
        )
        return events, history

    def test_identical_duplicates_are_collapsed(self):
        events, history = self._frames(conflicting=False)
        out, evidence = as_of_join(events, history, spec())
        assert out.loc[0, "v"] == 1
        assert len(out) == 1
        assert evidence.n_duplicate_rows_collapsed == 1

    def test_conflicting_duplicates_error_by_default(self):
        events, history = self._frames(conflicting=True)
        with pytest.raises(CovariateJoinError, match="duplicate the participant"):
            as_of_join(events, history, spec())

    @pytest.mark.parametrize("policy, expected", [("first", 1), ("last", 2)])
    def test_first_and_last_are_input_order(self, policy, expected):
        events, history = self._frames(conflicting=True)
        out, evidence = as_of_join(events, history, spec(on_duplicate=policy))
        assert out.loc[0, "v"] == expected
        assert evidence.n_duplicate_conflicts == 1

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(CovariateJoinError, match="Unknown on_duplicate"):
            CovariateJoinSpec(
                event_time="t", effective_time="eff", participant_key="pid",
                on_duplicate="whatever",
            )


# ---------------------------------------------------------------------------
# Missing policy
# ---------------------------------------------------------------------------


class TestMissingPolicy:
    def _frames(self):
        events = pd.DataFrame(
            {"pid": ["a", "b"], "t": pd.to_datetime(["2020-01-15", "2020-01-15"])},
            index=[10, 20],
        )
        history = pd.DataFrame(
            {
                "pid": ["a"],
                "eff": pd.to_datetime(["2020-01-01"]),
                "exp": pd.to_datetime(["2020-02-01"]),
                "v": [1],
            }
        )
        return events, history

    def test_error_is_the_default(self):
        events, history = self._frames()
        with pytest.raises(CovariateJoinError, match="no usable covariate record"):
            as_of_join(
                events, history,
                CovariateJoinSpec(
                    event_time="t", effective_time="eff", expires_time="exp",
                    participant_key="pid",
                ),
            )

    def test_drop_event_removes_rows_and_keeps_remaining_identity(self):
        events, history = self._frames()
        out, evidence = as_of_join(events, history, spec(on_missing="drop_event"))
        assert list(out.index) == [10]
        assert out["pid"].tolist() == ["a"]
        assert evidence.n_dropped == 1
        assert evidence.n_events_out == 1
        assert len(events) == 2  # caller frame unchanged

    def test_allow_na_keeps_rows_with_missing_attributes(self):
        events, history = self._frames()
        out, evidence = as_of_join(events, history, spec(on_missing="allow_na"))
        assert list(out.index) == [10, 20]
        assert pd.isna(out.loc[20, "v"])
        assert out.loc[20, "__tea_join_status"] == "missing"
        assert evidence.n_missing == 1

    def test_null_event_time_and_null_key_are_missing(self):
        events = pd.DataFrame(
            {
                "pid": ["a", None],
                "t": pd.to_datetime(["2020-01-15", None]),
            }
        )
        history = pd.DataFrame(
            {
                "pid": ["a", "a"],
                "eff": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "exp": pd.to_datetime(["2020-02-01", "2020-02-01"]),
                "v": [1, 1],
            }
        )
        out, evidence = as_of_join(events, history, spec(on_missing="allow_na"))
        assert out["__tea_join_status"].tolist() == ["matched", "missing"]
        assert evidence.n_null_event_key == 1
        assert evidence.n_invalid_event_time == 1

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(CovariateJoinError, match="Unknown on_missing"):
            CovariateJoinSpec(
                event_time="t", effective_time="eff", participant_key="pid",
                on_missing="maybe",
            )


# ---------------------------------------------------------------------------
# Declaration errors and degenerate inputs
# ---------------------------------------------------------------------------


class TestDeclarationErrors:
    def test_participant_key_is_required(self):
        with pytest.raises(CovariateJoinError, match="participant_key"):
            as_of_join(
                make_events(), make_history(),
                CovariateJoinSpec(event_time="t", effective_time="eff"),
            )

    def test_missing_column_names_the_frame_and_available_columns(self):
        with pytest.raises(CovariateJoinError, match="does not exist in the event table"):
            as_of_join(
                make_events(), make_history(),
                CovariateJoinSpec(
                    event_time="nope", effective_time="eff", participant_key="pid",
                ),
            )

    def test_expiry_must_be_after_effective(self):
        history = pd.DataFrame(
            {
                "pid": ["a"],
                "eff": pd.to_datetime(["2020-02-01"]),
                "exp": pd.to_datetime(["2020-01-01"]),
                "v": [1],
            }
        )
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-01-15"])})
        with pytest.raises(CovariateJoinError, match="expire strictly after"):
            as_of_join(events, history, spec())

    def test_at_least_one_attribute_is_required(self):
        history = pd.DataFrame(
            {"pid": ["a"], "eff": pd.to_datetime(["2020-01-01"])}
        )
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-01-15"])})
        with pytest.raises(CovariateJoinError, match="at least one attribute"):
            as_of_join(
                events, history,
                CovariateJoinSpec(event_time="t", effective_time="eff", participant_key="pid"),
            )

    def test_metadata_column_cannot_also_be_an_attribute(self):
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-01-15"])})
        history = pd.DataFrame(
            {"pid": ["a"], "eff": pd.to_datetime(["2020-01-01"]), "v": [1]}
        )
        with pytest.raises(CovariateJoinError, match="join metadata column"):
            as_of_join(
                events, history,
                CovariateJoinSpec(
                    event_time="t", effective_time="eff", participant_key="pid",
                    attributes=["pid"],
                ),
            )

    def test_duplicate_history_column_names_are_rejected(self):
        history = make_history()
        history.columns = ["pid", "eff", "exp", "grp", "grp"]
        with pytest.raises(CovariateJoinError, match="duplicate column names"):
            as_of_join(make_events(), history, spec())

    def test_reserved_prefix_is_rejected(self):
        with pytest.raises(CovariateJoinError, match="reserved prefix"):
            as_of_join(
                make_events(), make_history(), spec(prefix="__tea_join_")
            )

    def test_empty_frames_are_handled(self):
        empty_history = pd.DataFrame(
            {
                "pid": pd.Series([], dtype=object),
                "eff": pd.to_datetime([]),
                "v": pd.Series([], dtype=float),
            }
        )
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-01-15"])})
        out, evidence = as_of_join(
            events, empty_history, spec(expires_time=None)
        )
        assert len(out) == 1
        assert evidence.n_missing == 1

        empty_events = pd.DataFrame(
            {"pid": pd.Series([], dtype=object), "t": pd.to_datetime([])}
        )
        out, evidence = as_of_join(empty_events, make_history(), spec())
        assert len(out) == 0
        assert evidence.n_events_out == 0

    def test_non_string_participant_keys_are_supported(self):
        events = pd.DataFrame(
            {"pid": [1, 1, 2], "t": pd.to_datetime(["2020-02-01"] * 3)}
        )
        history = pd.DataFrame(
            {
                "pid": [1, 2],
                "eff": pd.to_datetime(["2020-01-01"] * 2),
                "v": [10, 20],
            }
        )
        out, _ = as_of_join(
            events, history, spec(expires_time=None)
        )
        assert out["v"].tolist() == [10, 10, 20]

    def test_null_history_key_is_ignored_with_a_warning(self):
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-02-01"])})
        history = pd.DataFrame(
            {
                "pid": ["a", None],
                "eff": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "v": [1, 2],
            }
        )
        out, evidence = as_of_join(
            events, history, spec(expires_time=None)
        )
        assert out["v"].tolist() == [1]
        assert any("null participant key" in w for w in evidence.warnings)


# ---------------------------------------------------------------------------
# Identity, evidence and snapshot
# ---------------------------------------------------------------------------


class TestIdentityAndEvidence:
    def test_event_count_never_grows(self):
        events = make_events()
        history = make_history()
        out, _ = as_of_join(events, history, spec())
        assert len(out) == len(events)

    def test_row_order_and_index_are_preserved(self):
        events = make_events()
        events.index = pd.Index([100, 101, 102, 103, 104, 105], name="row")
        out, _ = as_of_join(events, history := make_history(), spec())
        assert list(out.index) == [100, 101, 102, 103, 104, 105]
        assert out.index.name == "row"
        assert out["eid"].tolist() == events["eid"].tolist()

    def test_evidence_identifies_the_source_row_per_event(self):
        out, evidence = as_of_join(make_events(), make_history(), spec())
        by_event = out.set_index("eid")
        assert by_event.loc[1, "__tea_join_matched_row"] == 0
        assert by_event.loc[2, "__tea_join_matched_row"] == 1
        assert by_event.loc[3, "__tea_join_matched_row"] == 2
        assert pd.isna(by_event.loc[5, "__tea_join_matched_row"])
        assert evidence.record_effective_time.iloc[0] == pd.Timestamp("2020-01-01")
        assert evidence.record_expires_time.iloc[1] == pd.Timestamp("2020-03-15")

    def test_evidence_as_dict_round_trips(self):
        _, evidence = as_of_join(
            make_events(), make_history(), spec(), covariate_variables()
        )
        payload = evidence.as_dict()
        assert payload["n_events_in"] == payload["n_events_out"] == 6
        assert payload["policies"]["on_missing"] == "allow_na"
        assert "age" in payload["validated_attributes"]

    def test_prefix_avoids_footprint_in_source_frame(self):
        events = make_events()
        history = make_history()
        out, _ = as_of_join(events, history, spec(prefix="base_"))
        assert "base_grp" in out.columns
        assert "grp" not in events.columns
        assert list(events.columns) == ["eid", "pid", "t", "y"]

    def test_duplicate_attribute_in_event_frame_is_rejected(self):
        events = make_events().assign(grp="existing")
        with pytest.raises(CovariateJoinError, match="already exists"):
            as_of_join(events, make_history(), spec())

    def test_snapshot_ties_data_to_schema(self):
        variables = covariate_variables()
        out, evidence = as_of_join(make_events(), make_history(), spec(), variables)
        snapshot = make_snapshot(out, variables, is_joined=True, evidence=evidence)
        assert snapshot.is_joined
        assert snapshot.evidence is evidence
        assert snapshot.schema["grp"]["dtype"] is DataType.NOMINAL
        snapshot.assert_current()  # no error

        snapshot.data.iloc[0, snapshot.data.columns.get_loc("y")] = -1
        with pytest.raises(CovariateJoinError, match="changed after the covariate join"):
            snapshot.assert_current()


# ---------------------------------------------------------------------------
# Schema validation of attached attributes
# ---------------------------------------------------------------------------


class TestAttributeValidation:
    def test_valid_attributes_are_reported(self):
        out, evidence = as_of_join(
            make_events(), make_history(), spec(), covariate_variables()
        )
        assert evidence.validated_attributes == {"age": "ok", "grp": "ok"}

    def test_out_of_schema_category_is_rejected(self):
        history = make_history().assign(grp=["x", "z", "x"])
        with pytest.raises(CovariateJoinError, match="outside the declared nominal"):
            as_of_join(make_events(), history, spec(), covariate_variables())

    def test_out_of_range_numeric_is_rejected(self):
        variables = [nominal("grp", ["x", "y"]), ratio("age", [0, 40])]
        with pytest.raises(CovariateJoinError, match="outside its declared range"):
            as_of_join(make_events(), make_history(), spec(), variables)

    def test_non_numeric_value_for_numeric_attribute_is_rejected(self):
        history = make_history().astype({"age": object}).assign(
            age=["old", "new", "old"]
        )
        with pytest.raises(CovariateJoinError, match="non-numeric"):
            as_of_join(make_events(), history, spec(), covariate_variables())

    def test_undeclared_attribute_is_not_revalidated(self):
        report = validate_attribute_schema(
            make_history(), [], ["grp", "age"]
        )
        assert report == {"grp": "undeclared", "age": "undeclared"}


# ---------------------------------------------------------------------------
# Time precision and time zone
# ---------------------------------------------------------------------------


class TestTimeAgreement:
    def test_precision_mismatch_is_rejected(self):
        events = pd.DataFrame(
            {"pid": ["a"], "t": pd.Series(pd.to_datetime(["2020-01-15"]).astype("datetime64[s]"))}
        )
        history = make_history()
        with pytest.raises(CovariateJoinError, match="disagree on precision"):
            as_of_join(events, history, spec())

    def test_precision_can_be_declared(self):
        events = pd.DataFrame(
            {"pid": ["a"], "t": pd.Series(pd.to_datetime(["2020-01-15"]).astype("datetime64[us]"))}
        )
        out, _ = as_of_join(events, make_history(), spec(time_unit="us"))
        assert out.loc[0, "grp"] == "x"

    def test_time_zone_mismatch_is_rejected(self):
        events = pd.DataFrame(
            {"pid": ["a"], "t": pd.to_datetime(["2020-01-15"]).tz_localize("UTC")}
        )
        history = make_history().assign(
            eff=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-10"]).tz_localize(
                "America/New_York"
            ),
            exp=pd.to_datetime(["2020-02-01", "2020-03-15", "2020-02-01"]).tz_localize(
                "America/New_York"
            ),
        )
        with pytest.raises(CovariateJoinError, match="disagree on time zone"):
            as_of_join(events, history, spec())

    def test_naive_and_aware_cannot_be_mixed(self):
        events = pd.DataFrame({"pid": ["a"], "t": pd.to_datetime(["2020-01-15"])})
        history = make_history().assign(
            eff=pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-01-10"]
            ).tz_localize("UTC"),
            exp=pd.to_datetime(
                ["2020-02-01", "2020-03-15", "2020-02-01"]
            ).tz_localize("UTC"),
        )
        with pytest.raises(CovariateJoinError, match="disagree on time zone"):
            as_of_join(events, history, spec())

    def test_declared_time_zone_aligns_columns(self):
        events = pd.DataFrame(
            {"pid": ["a"], "t": pd.to_datetime(["2020-01-15"]).tz_localize("UTC")}
        )
        history = make_history().assign(
            eff=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-10"]),
            exp=pd.to_datetime(["2020-02-01", "2020-03-15", "2020-02-01"]),
        )
        out, _ = as_of_join(events, history, spec(time_zone="UTC"))
        assert out.loc[0, "grp"] == "x"

    def test_declared_time_zone_converts_aware_columns(self):
        events = pd.DataFrame(
            {"pid": ["a"], "t": pd.to_datetime(["2020-01-15 12:00"]).tz_localize("UTC")}
        )
        history = make_history().assign(
            eff=pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-01-10"]
            ).tz_localize("America/New_York"),
            exp=pd.to_datetime(
                ["2020-02-01", "2020-03-15", "2020-02-01"]
            ).tz_localize("America/New_York"),
        )
        out, _ = as_of_join(events, history, spec(time_zone="UTC"))
        # Same instant, expressed in UTC; still the 01-01 version.
        assert out.loc[0, "grp"] == "x"

    def test_lossy_precision_conversion_is_rejected(self):
        events = pd.DataFrame(
            {"pid": ["a"], "t": pd.to_datetime(["2020-01-15T00:00:00.5"])}
        )
        with pytest.raises(CovariateJoinError, match="change recorded instants"):
            as_of_join(events, make_history(), spec(time_unit="s"))


# ---------------------------------------------------------------------------
# Dataset entry point
# ---------------------------------------------------------------------------


class TestDatasetJoin:
    def test_join_returns_new_dataset_and_leaves_inputs_untouched(self):
        events = make_events()
        history = make_history()
        original_events = events.copy(deep=True)
        original_history = history.copy(deep=True)

        dataset = Dataset(events, covariate_variables(), "pid")
        joined = dataset.join_covariates(history, spec())

        pd.testing.assert_frame_equal(events, original_events)
        pd.testing.assert_frame_equal(history, original_history)
        assert joined is not dataset
        assert list(joined.data.columns)[:4] == ["eid", "pid", "t", "y"]
        assert joined.evidence is not None
        assert joined.snapshot.is_joined
        joined.assert_snapshot_current()

    def test_inplace_join_updates_snapshot(self):
        dataset = Dataset(make_events(), covariate_variables(), "pid")
        returned = dataset.join_covariates(make_history(), spec(), inplace=True)
        assert returned is dataset
        assert "grp" in dataset.data.columns
        assert dataset.snapshot.is_joined
        dataset.assert_snapshot_current()

    def test_join_does_not_inflate_sample_size(self):
        dataset = Dataset(make_events(), covariate_variables(), "pid")
        joined = dataset.join_covariates(make_history(), spec())
        assert len(joined.data) == len(dataset.data)

    def test_history_can_be_a_dataset(self):
        events_ds = Dataset(make_events(), covariate_variables(), "pid")
        history_ds = Dataset(make_history(), [], "pid")
        joined = events_ds.join_covariates(history_ds, spec())
        assert joined.data["grp"].notna().sum() == 3


# ---------------------------------------------------------------------------
# End-to-end through the public API
# ---------------------------------------------------------------------------


def _fresh_analysis(history, spec_obj, variables, design, events=None):
    events = make_events() if events is None else events
    tea.data(events, key="pid")
    tea.define_variables(variables)
    tea.define_study_design(design)
    tea.assume({"Type I (False Positive) Error Rate": 0.05})
    tea.attach_covariates(history, spec_obj)
    return events


def _analysis_frames(seed=0, n_per_group=30):
    """Larger, deterministic event/history pair with a real group difference.

    The joined attribute ``grp`` separates the two groups; ``y`` is shifted so
    tests produce a non-degenerate statistic.
    """
    rng = np.random.RandomState(seed)
    n = n_per_group * 2
    pids = [f"p{i}" for i in range(n)]
    groups = ["x"] * n_per_group + ["y"] * n_per_group
    y = rng.normal(0.0, 1.0, size=n)
    y[n_per_group:] += 2.0
    events = pd.DataFrame(
        {
            "pid": pids,
            "t": pd.to_datetime(["2020-02-15"] * n),
            "y": y,
        }
    )
    history = pd.DataFrame(
        {
            "pid": pids,
            "eff": pd.to_datetime(["2020-01-01"] * n),
            "exp": pd.to_datetime(["2020-03-01"] * n),
            "grp": groups,
        }
    )
    return events, history


class TestApiIntegration:
    def test_joined_attribute_can_be_hypothesized(self):
        events, history = _analysis_frames()
        variables = [
            {"name": "grp", "data type": "nominal", "categories": ["x", "y"]},
            {"name": "y", "data type": "ratio"},
        ]
        design = {
            "study type": "experiment",
            "independent variables": "grp",
            "dependent variables": "y",
        }
        _fresh_analysis(history, spec(), variables, design, events=events)

        result = tea.hypothesize(["grp", "y"], ["grp:x != y"])
        evidence = tea.get_covariate_evidence()

        assert result is not None
        assert evidence.n_events_out == len(events)
        assert evidence.n_matched == len(events)
        assert evidence.n_missing == 0

    def test_analysis_table_is_not_modified_and_sample_size_is_stable(self):
        events, history = _analysis_frames()
        variables = [
            {"name": "grp", "data type": "nominal", "categories": ["x", "y"]},
            {"name": "y", "data type": "ratio"},
        ]
        design = {
            "study type": "experiment",
            "independent variables": "grp",
            "dependent variables": "y",
        }
        _fresh_analysis(history, spec(), variables, design, events=events)
        before = events.copy(deep=True)

        tea.hypothesize(["grp", "y"], ["grp:x != y"])

        pd.testing.assert_frame_equal(events, before)
        assert tea.get_covariate_evidence().n_events_in == len(before)
        assert tea.get_covariate_evidence().n_events_out == len(before)

    def test_schema_violation_surfaces_at_hypothesize(self):
        events, history = _analysis_frames()
        bad_history = history.assign(grp=["x"] * len(history))
        bad_history.loc[0, "grp"] = "nope"
        variables = [
            {"name": "grp", "data type": "nominal", "categories": ["x", "y"]},
            {"name": "y", "data type": "ratio"},
        ]
        design = {
            "study type": "experiment",
            "independent variables": "grp",
            "dependent variables": "y",
        }
        _fresh_analysis(bad_history, spec(), variables, design, events=events)

        with pytest.raises(CovariateJoinError, match="outside the declared nominal"):
            tea.hypothesize(["grp", "y"], ["grp:x != y"])

    def test_data_call_clears_previous_history(self):
        events, history = _analysis_frames()
        variables = [
            {"name": "grp", "data type": "nominal", "categories": ["x", "y"]},
            {"name": "y", "data type": "ratio"},
        ]
        design = {
            "study type": "experiment",
            "independent variables": "grp",
            "dependent variables": "y",
        }
        _fresh_analysis(history, spec(), variables, design, events=events)

        # Declaring a new dataset must not silently reuse the old history.
        from tea import api as tea_api

        tea.data(events, key="pid")
        assert tea_api.covariate_history is None
        assert tea_api.covariate_spec is None

    def test_attach_covariates_requires_a_spec(self):
        tea.data(make_events(), key="pid")
        with pytest.raises(ValueError, match="CovariateJoinSpec"):
            tea.attach_covariates(make_history(), {"event_time": "t"})
