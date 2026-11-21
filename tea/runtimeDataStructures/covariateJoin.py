"""Temporal (as-of) covariate joins for the Tea runtime.

The event-level analysis table and a participant-keyed history table are
connected by declaring three things:

* the event-time column in the analysis table,
* the time at which a history record becomes effective,
* optionally, the time at which it stops being valid.

Every event receives **at most one** history record: the one that was in force
at the event time.  Records that become effective *after* an event are never
used to backfill it, so no information from the future leaks into the past.

Three ambiguous situations must be resolved explicitly:

``on_overlap``
    Two versions of the same attribute are valid at the same instant.
    ``'error'`` (default), ``'latest_effective'`` (the most recently effective
    version wins; the earlier version is truncated) or ``'earliest_effective'``
    (the older still-valid version wins; the later version starts only once the
    older one expires).

``on_duplicate``
    Two history rows share the same ``(participant, effective_time)`` key.
    ``'error'`` (default), ``'first'`` or ``'last'`` (caller input order).
    Rows that are identical over the attached attributes and expiry are
    collapsed silently and reported in the join evidence.

``on_missing``
    An event has no record in force at its event time (before the first
    version, inside a gap left by an explicit expiry, a null event time, or a
    null participant key).  ``'error'`` (default), ``'drop_event'`` or
    ``'allow_na'``.

All three are validated at join time; an unknown value is an error rather than
a silent fallback.

The join preserves the event rows' identity (same order, same index) and
records, per event, which history row was used and why.  The result is a
single :class:`AnalysisSnapshot`: the variable schema, attribute validation
and ``hypothesize`` all read the same frame.  History rows are only ever
looked up, never merged one-to-many, so the event count can decrease under
``on_missing='drop_event'`` but can never grow.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from tea.ast import DataType

__all__ = [
    "CovariateJoinError",
    "OnOverlap",
    "OnDuplicate",
    "OnMissing",
    "CovariateJoinSpec",
    "JoinEvidence",
    "AnalysisSnapshot",
    "as_of_join",
    "validate_attribute_schema",
    "schema_from_variables",
    "fingerprint_frame",
    "JOIN_EVIDENCE_PREFIX",
]

# Columns added to the analysis frame.  User columns may not start with this
# prefix, so the evidence can never overwrite user data.
JOIN_EVIDENCE_PREFIX = "__tea_join_"

_OPEN_ENDED = int(np.iinfo(np.int64).max)
_NEG_INF = int(np.iinfo(np.int64).min)


class CovariateJoinError(ValueError):
    """Raised when a temporal covariate join cannot be performed as declared."""


class _StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class OnOverlap(_StrEnum):
    ERROR = "error"
    LATEST_EFFECTIVE = "latest_effective"
    EARLIEST_EFFECTIVE = "earliest_effective"


class OnDuplicate(_StrEnum):
    ERROR = "error"
    FIRST = "first"
    LAST = "last"


class OnMissing(_StrEnum):
    ERROR = "error"
    DROP_EVENT = "drop_event"
    ALLOW_NA = "allow_na"


def _coerce_policy(value: Union[str, _StrEnum], enum_cls, what: str):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError:
        choices = ", ".join(repr(member.value) for member in enum_cls)
        raise CovariateJoinError(
            f"Unknown {what} policy {value!r}. Expected one of: {choices}."
        ) from None


@dataclass(frozen=True)
class CovariateJoinSpec:
    """Declarative description of one event/history temporal join."""

    event_time: str
    effective_time: str
    expires_time: Optional[str] = None
    participant_key: Optional[str] = None
    attributes: Optional[Tuple[str, ...]] = None
    prefix: str = ""
    on_overlap: Union[str, OnOverlap] = OnOverlap.ERROR
    on_duplicate: Union[str, OnDuplicate] = OnDuplicate.ERROR
    on_missing: Union[str, OnMissing] = OnMissing.ERROR
    time_zone: Optional[str] = None
    time_unit: Optional[str] = None
    validate: bool = True

    def __post_init__(self):
        attrs = self.attributes
        if attrs is not None:
            object.__setattr__(self, "attributes", tuple(attrs))
        object.__setattr__(
            self, "on_overlap", _coerce_policy(self.on_overlap, OnOverlap, "on_overlap")
        )
        object.__setattr__(
            self,
            "on_duplicate",
            _coerce_policy(self.on_duplicate, OnDuplicate, "on_duplicate"),
        )
        object.__setattr__(
            self, "on_missing", _coerce_policy(self.on_missing, OnMissing, "on_missing")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_time": self.event_time,
            "effective_time": self.effective_time,
            "expires_time": self.expires_time,
            "participant_key": self.participant_key,
            "attributes": list(self.attributes) if self.attributes else None,
            "prefix": self.prefix,
            "on_overlap": self.on_overlap.value,
            "on_duplicate": self.on_duplicate.value,
            "on_missing": self.on_missing.value,
            "time_zone": self.time_zone,
            "time_unit": self.time_unit,
        }


@dataclass
class JoinEvidence:
    """Per-join audit trail: what was matched, what was ambiguous, what was dropped."""

    n_events_in: int = 0
    n_events_out: int = 0
    n_matched: int = 0
    n_missing: int = 0
    n_dropped: int = 0
    n_invalid_event_time: int = 0
    n_null_event_key: int = 0
    n_null_attribute_values: int = 0
    n_duplicate_rows_collapsed: int = 0
    n_duplicate_conflicts: int = 0
    n_overlaps_resolved: int = 0
    n_history_rows_in: int = 0
    n_history_rows_used: int = 0
    policies: Dict[str, str] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    attributes: Tuple[str, ...] = ()
    matched_history_rows: Optional[pd.Series] = None
    event_time: Optional[pd.Series] = None
    record_effective_time: Optional[pd.Series] = None
    record_expires_time: Optional[pd.Series] = None
    per_event_status: Optional[pd.Series] = None
    source_fingerprint: str = ""
    result_fingerprint: str = ""
    validated_attributes: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Covariate join evidence",
            "------------------------",
            f"events in: {self.n_events_in}",
            f"events out: {self.n_events_out}",
            f"matched: {self.n_matched}",
            f"missing: {self.n_missing} (dropped: {self.n_dropped})",
            f"history rows in: {self.n_history_rows_in}",
            f"history rows used: {self.n_history_rows_used}",
            f"duplicate rows collapsed: {self.n_duplicate_rows_collapsed}",
            f"duplicate conflicts: {self.n_duplicate_conflicts}",
            f"overlapping versions resolved: {self.n_overlaps_resolved}",
            "policies: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.policies.items())),
            f"source fingerprint: {self.source_fingerprint[:16]}",
            f"result fingerprint: {self.result_fingerprint[:16]}",
        ]
        if self.validated_attributes:
            lines.append(
                "attribute validation: "
                + ", ".join(
                    f"{k}={v}" for k, v in sorted(self.validated_attributes.items())
                )
            )
        for warning in self.warnings:
            lines.append(f"warning: {warning}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_events_in": self.n_events_in,
            "n_events_out": self.n_events_out,
            "n_matched": self.n_matched,
            "n_missing": self.n_missing,
            "n_dropped": self.n_dropped,
            "n_invalid_event_time": self.n_invalid_event_time,
            "n_null_event_key": self.n_null_event_key,
            "n_null_attribute_values": self.n_null_attribute_values,
            "n_duplicate_rows_collapsed": self.n_duplicate_rows_collapsed,
            "n_duplicate_conflicts": self.n_duplicate_conflicts,
            "n_overlaps_resolved": self.n_overlaps_resolved,
            "n_history_rows_in": self.n_history_rows_in,
            "n_history_rows_used": self.n_history_rows_used,
            "policies": dict(self.policies),
            "spec": dict(self.spec),
            "attributes": list(self.attributes),
            "validated_attributes": dict(self.validated_attributes),
            "warnings": list(self.warnings),
            "source_fingerprint": self.source_fingerprint,
            "result_fingerprint": self.result_fingerprint,
        }


@dataclass(frozen=True, eq=False)
class AnalysisSnapshot:
    """One immutable view of the data together with its variable schema.

    ``data``, ``schema``/``variables`` and the join evidence always describe
    the same frame.  Consumers (schema validation, ``hypothesize``) share this
    object, so they cannot accidentally read a pre-join frame.
    """

    data: pd.DataFrame
    variables: Tuple[Any, ...]
    schema: Mapping[str, Dict[str, Any]]
    fingerprint: str
    is_joined: bool = False
    evidence: Optional[JoinEvidence] = None

    def assert_current(self) -> None:
        """Raise if ``data`` no longer matches the snapshot fingerprint."""
        if fingerprint_frame(self.data) != self.fingerprint:
            raise CovariateJoinError(
                "The analysis data changed after the covariate join. "
                "Schema, attribute validation and hypothesize must all consume "
                "the same post-join snapshot; rebuild the Dataset instead of "
                "mutating .data in place."
            )


def schema_from_variables(variables: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    schema: Dict[str, Dict[str, Any]] = {}
    for var in variables or ():
        categories = getattr(var, "categories", None)
        schema[var.name] = {
            "dtype": getattr(var, "dtype", None),
            "categories": categories,
            "drange": getattr(var, "drange", None),
        }
    return schema


def fingerprint_frame(data: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr(list(data.columns)).encode("utf-8"))
    digest.update(repr([str(dtype) for dtype in data.dtypes]).encode("utf-8"))
    digest.update(repr(tuple(data.index)).encode("utf-8"))
    digest.update(repr(data.index.name).encode("utf-8"))
    try:
        hashed = pd.util.hash_pandas_object(data, index=False).to_numpy()
        digest.update(hashed.tobytes())
    except Exception:  # pragma: no cover - exotic object columns
        digest.update(repr(data.shape).encode("utf-8"))
        for column in data.columns:
            digest.update(repr(data[column].tolist()).encode("utf-8"))
    return digest.hexdigest()


def fingerprint_variables(variables: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for var in variables or ():
        digest.update(
            repr(
                (
                    getattr(var, "name", None),
                    getattr(getattr(var, "dtype", None), "name", None),
                    repr(getattr(var, "categories", None)),
                    repr(getattr(var, "drange", None)),
                )
            ).encode("utf-8")
        )
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------


def _is_datetime_like(dtype) -> bool:
    if isinstance(dtype, pd.DatetimeTZDtype):
        return True
    try:
        return bool(pd.api.types.is_datetime64_dtype(dtype))
    except Exception:  # pragma: no cover - defensive
        return False


def _time_unit(dtype) -> str:
    if isinstance(dtype, pd.DatetimeTZDtype):
        return str(dtype.unit)
    return str(np.datetime_data(dtype)[0])


def _time_zone(dtype) -> Optional[str]:
    if isinstance(dtype, pd.DatetimeTZDtype):
        return str(dtype.tz)
    return None


def _coerce_time(
    series: pd.Series,
    column: str,
    role: str,
    *,
    time_zone: Optional[str],
    time_unit: Optional[str],
) -> pd.Series:
    """Parse ``series`` to datetimes, applying the join's declared tz/unit."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if _is_datetime_like(series.dtype):
        parsed = series
    else:
        # pandas warns (and will later raise) on mixed-offset strings; we handle
        # that case explicitly below, so keep the warning out of user output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                parsed = pd.to_datetime(series, errors="raise")
            except Exception as exc:
                raise CovariateJoinError(
                    f"{role} column {column!r} could not be parsed as datetimes: {exc}"
                ) from None
            if not _is_datetime_like(parsed.dtype):
                # Mixed timezone offsets parse to object dtype in pandas 2.x.
                try:
                    parsed = pd.to_datetime(series, errors="raise", utc=True)
                except Exception as exc:
                    raise CovariateJoinError(
                        f"{role} column {column!r} contains ambiguous or mixed "
                        f"time zones and could not be parsed: {exc}. "
                        "Specify time_zone= to state the intended zone."
                    ) from None
        if not _is_datetime_like(parsed.dtype):
            raise CovariateJoinError(
                f"{role} column {column!r} contains mixed time zones; "
                "specify time_zone= explicitly."
            )

    if time_zone is not None:
        if _time_zone(parsed.dtype) is None:
            try:
                parsed = parsed.dt.tz_localize(time_zone)
            except Exception as exc:
                raise CovariateJoinError(
                    f"{role} column {column!r} has a timestamp that is ambiguous or "
                    f"does not exist in time zone {time_zone!r} "
                    f"(daylight-saving transition): {exc}. "
                    "Disambiguate those timestamps before joining."
                ) from None
        else:
            parsed = parsed.dt.tz_convert(time_zone)

    if time_unit is not None:
        try:
            parsed = parsed.dt.as_unit(time_unit)
        except (AttributeError, ValueError) as exc:
            raise CovariateJoinError(
                f"Could not convert {role} column {column!r} to time unit "
                f"{time_unit!r}: {exc}"
            ) from None

    return parsed.reset_index(drop=True)


def _assert_time_compatible(
    columns: Dict[str, pd.Series], spec: CovariateJoinSpec
) -> None:
    units = {_time_unit(series.dtype) for series in columns.values()}
    zones = {_time_zone(series.dtype) for series in columns.values()}
    if len(units) > 1:
        detail = ", ".join(
            f"{name}={_time_unit(series.dtype)!r}" for name, series in columns.items()
        )
        raise CovariateJoinError(
            "Time columns disagree on precision ("
            + detail
            + "). Convert them to one unit, or declare time_unit= "
            f"(for example time_unit='{sorted(units, key=len)[0]}')."
        )
    if len(zones) > 1:
        detail = ", ".join(
            f"{name}={_time_zone(series.dtype)!r}" for name, series in columns.items()
        )
        raise CovariateJoinError(
            "Time columns disagree on time zone ("
            + detail
            + "). Convert them to one zone, or declare time_zone=."
        )


def _time_to_int(series: pd.Series) -> np.ndarray:
    return series.astype("int64").to_numpy(dtype=np.int64, copy=False)


# ---------------------------------------------------------------------------
# Schema validation of attached attributes
# ---------------------------------------------------------------------------


def _category_contains(categories: Mapping[Any, Any], value: Any) -> bool:
    try:
        if value in categories:
            return True
    except TypeError:  # pragma: no cover - unhashable category value
        return False
    # Tolerate numpy scalar / python scalar and string/number round-tripping.
    for candidate in categories:
        if candidate == value:
            return True
        try:
            if type(candidate) is not type(value) and str(candidate) == str(value):
                return True
        except Exception:  # pragma: no cover - defensive
            continue
    return False


def _find_variable(variables: Sequence[Any], name: str):
    for var in variables or ():
        if getattr(var, "name", None) == name:
            return var
    return None


def validate_attribute_schema(
    data: pd.DataFrame,
    variables: Sequence[Any],
    attribute_names: Sequence[str],
) -> Dict[str, str]:
    """Validate attached columns against the declared variable schema.

    Only attached columns are checked.  Declared variables that do not come
    from the history table keep their historical behaviour (they are not
    re-validated here), so this cannot change existing analyses.

    Returns a per-attribute report.  Raises :class:`CovariateJoinError` on a
    violation, naming the column, the variable kind, and a sample of offending
    values.
    """
    report: Dict[str, str] = {}
    for name in attribute_names:
        var = _find_variable(variables, name)
        if var is None:
            report[name] = "undeclared"
            continue

        series = data[name]
        non_null = series.dropna()

        if var.dtype is DataType.NOMINAL or var.dtype is DataType.ORDINAL:
            categories = var.categories or {}
            if len(non_null) == 0:
                report[name] = "empty"
                continue
            bad = [
                value
                for value in pd.unique(non_null)
                if not _category_contains(categories, value)
            ]
            if bad:
                raise CovariateJoinError(
                    f"Attached attribute {name!r} has values outside the declared "
                    f"{var.dtype.name.lower()} categories: {bad[:5]!r}. "
                    f"Declared categories: {list(categories)!r}."
                )
            report[name] = "ok"
        elif var.dtype is DataType.INTERVAL or var.dtype is DataType.RATIO:
            numeric = pd.to_numeric(non_null, errors="coerce")
            invalid = non_null[numeric.isna()]
            if len(invalid):
                raise CovariateJoinError(
                    f"Attached attribute {name!r} is declared "
                    f"{var.dtype.name.lower()} but contains non-numeric values: "
                    f"{list(pd.unique(invalid))[:5]!r}."
                )
            drange = getattr(var, "drange", None)
            if drange is not None and len(numeric):
                low, high = drange
                out_of_range = non_null[(numeric < low) | (numeric > high)]
                if len(out_of_range):
                    raise CovariateJoinError(
                        f"Attached attribute {name!r} has values outside its "
                        f"declared range [{low!r}, {high!r}]: "
                        f"{list(pd.unique(out_of_range))[:5]!r}."
                    )
            report[name] = "ok"
        else:
            report[name] = "unchecked"

    return report


# ---------------------------------------------------------------------------
# Attribute taking / equality helpers
# ---------------------------------------------------------------------------


def _take_optional(series: pd.Series, positions: np.ndarray) -> pd.Series:
    """Take ``positions`` positionally; a negative position yields a missing value.

    Uses an ``Int64`` index so ``reindex`` inserts NA instead of raising, and
    keeps the original dtype where pandas can represent the missing value.
    """
    index = pd.Index(pd.Series(positions, dtype="Int64"))
    taken = series.reindex(index)
    taken.index = pd.RangeIndex(len(taken))
    return taken


def _same_scalar(left: Any, right: Any) -> bool:
    left_na = left is None or left is pd.NaT or (not isinstance(left, (list, tuple)) and pd.isna(left))
    right_na = right is None or right is pd.NaT or (not isinstance(right, (list, tuple)) and pd.isna(right))
    try:
        left_na = bool(left_na)
        right_na = bool(right_na)
    except (TypeError, ValueError):  # pragma: no cover - array-like values
        left_na = right_na = False
    if left_na or right_na:
        return left_na and right_na
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(left) == repr(right)


def _rows_identical(
    attributes: pd.DataFrame, positions: Sequence[int], expiry: pd.Series
) -> bool:
    first = positions[0]
    for position in positions[1:]:
        for column in attributes.columns:
            if not _same_scalar(
                attributes[column].iloc[first], attributes[column].iloc[position]
            ):
                return False
        if not _same_scalar(expiry.iloc[first], expiry.iloc[position]):
            return False
    return True


def _row_list(positions: Sequence[int]) -> str:
    positions = sorted(int(p) for p in positions)
    shown = positions[:5]
    suffix = "" if len(positions) <= 5 else f", ... ({len(positions)} total)"
    return ", ".join(str(p) for p in shown) + suffix


# ---------------------------------------------------------------------------
# Time conversion guards
# ---------------------------------------------------------------------------


def _assert_lossless_conversion(
    original: pd.Series, converted: pd.Series, column: str, role: str
) -> None:
    """Reject a tz/unit conversion that would move an instant."""
    original_unit = _time_unit(original.dtype)
    try:
        round_tripped = converted.dt.as_unit(original_unit)
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        return
    comparable = original.notna()
    if comparable.any():
        left = original[comparable]
        right = round_tripped[comparable]
        if not left.equals(right):
            raise CovariateJoinError(
                f"Converting {role} column {column!r} would change recorded "
                "instants (the declared time precision is coarser than the data). "
                "Round the timestamps explicitly before joining, or use a finer "
                "time_unit."
            )


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _require_column(frame: pd.DataFrame, column: Optional[str], what: str, frame_name: str):
    if column is None:
        raise CovariateJoinError(f"A {what} column must be declared.")
    if column not in frame.columns:
        raise CovariateJoinError(
            f"{what} column {column!r} does not exist in the {frame_name}. "
            f"Available columns: {list(frame.columns)!r}."
        )


def _resolve_attribute_columns(
    events: pd.DataFrame, history: pd.DataFrame, spec: CovariateJoinSpec
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    reserved = {spec.effective_time, spec.expires_time, spec.participant_key}
    if spec.attributes is None:
        sources = tuple(column for column in history.columns if column not in reserved)
    else:
        sources = tuple(spec.attributes)

    if not sources:
        raise CovariateJoinError(
            "A covariate join must attach at least one attribute column."
        )

    seen_sources = set()
    for source in sources:
        if source in seen_sources:
            raise CovariateJoinError(f"Attribute {source!r} is declared twice.")
        seen_sources.add(source)
        if source in reserved and source is not None:
            raise CovariateJoinError(
                f"{source!r} is a join metadata column (participant key or "
                "time column) and cannot also be attached as an attribute."
            )
        _require_column(history, source, "attribute", "history table")

    outputs = tuple(f"{spec.prefix}{source}" for source in sources)
    if len(set(outputs)) != len(outputs):
        raise CovariateJoinError(
            f"Attribute names collide after applying prefix {spec.prefix!r}: {outputs!r}."
        )
    for output in outputs:
        if output.startswith(JOIN_EVIDENCE_PREFIX):
            raise CovariateJoinError(
                f"Attribute {output!r} uses the reserved prefix "
                f"{JOIN_EVIDENCE_PREFIX!r}; choose another name."
            )
        if output in events.columns:
            raise CovariateJoinError(
                f"Attribute {output!r} already exists in the event table. "
                "Joining would overwrite caller data; choose a prefix."
            )
        if output == spec.event_time:
            raise CovariateJoinError(
                f"Attribute {output!r} collides with the declared event time column."
            )
    for column in events.columns:
        if isinstance(column, str) and column.startswith(JOIN_EVIDENCE_PREFIX):
            raise CovariateJoinError(
                f"Event table column {column!r} uses the reserved prefix "
                f"{JOIN_EVIDENCE_PREFIX!r}."
            )
    return sources, outputs


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


def as_of_join(
    events: pd.DataFrame,
    history: pd.DataFrame,
    spec: CovariateJoinSpec,
    variables: Optional[Sequence[Any]] = None,
) -> Tuple[pd.DataFrame, JoinEvidence]:
    """Join ``history`` attributes onto ``events`` as of each event time.

    ``events`` and ``history`` are never mutated.  The returned frame keeps the
    event rows' order and index (minus any rows dropped under
    ``on_missing='drop_event'``), so event identity survives the join.  At most
    one history row is attached per event, so the join can never inflate the
    event count.
    """
    if not isinstance(events, pd.DataFrame):
        raise CovariateJoinError(
            f"events must be a pandas DataFrame, received {type(events).__name__}."
        )
    if not isinstance(history, pd.DataFrame):
        raise CovariateJoinError(
            f"history must be a pandas DataFrame, received {type(history).__name__}."
        )
    if spec is None:
        raise CovariateJoinError("A CovariateJoinSpec is required.")
    if spec.participant_key is None:
        raise CovariateJoinError(
            "A participant_key column must be declared: the history table is "
            "scoped per participant, so every event must name its participant."
        )
    if not history.columns.is_unique:
        duplicated = history.columns[history.columns.duplicated()].tolist()
        raise CovariateJoinError(
            f"The history table has duplicate column names: {duplicated!r}; "
            "attribute semantics would be ambiguous."
        )

    _require_column(events, spec.event_time, "event time", "event table")
    _require_column(events, spec.participant_key, "participant key", "event table")
    _require_column(history, spec.effective_time, "effective time", "history table")
    _require_column(history, spec.participant_key, "participant key", "history table")
    if spec.expires_time is not None:
        _require_column(history, spec.expires_time, "expiry time", "history table")

    sources, outputs = _resolve_attribute_columns(events, history, spec)

    evidence = JoinEvidence(
        n_events_in=len(events),
        n_history_rows_in=len(history),
        spec=spec.to_dict(),
        policies={
            "on_overlap": spec.on_overlap.value,
            "on_duplicate": spec.on_duplicate.value,
            "on_missing": spec.on_missing.value,
        },
        attributes=outputs,
    )

    # -- parse and reconcile time columns -----------------------------------
    time_kwargs = {"time_zone": spec.time_zone, "time_unit": spec.time_unit}
    event_times = _coerce_time(
        events[spec.event_time], spec.event_time, "event time", **time_kwargs
    )
    effective_times = _coerce_time(
        history[spec.effective_time],
        spec.effective_time,
        "effective time",
        **time_kwargs,
    )
    if spec.expires_time is not None:
        expiry_times = _coerce_time(
            history[spec.expires_time], spec.expires_time, "expiry time", **time_kwargs
        )
    else:
        expiry_times = pd.Series(pd.NaT, index=pd.RangeIndex(len(history)), dtype=effective_times.dtype)

    if spec.time_unit is not None:
        for original, converted, column, role in (
            (events[spec.event_time], event_times, spec.event_time, "event time"),
            (
                history[spec.effective_time],
                effective_times,
                spec.effective_time,
                "effective time",
            ),
        ):
            if _is_datetime_like(original.dtype):
                _assert_lossless_conversion(original, converted, column, role)
        if spec.expires_time is not None and _is_datetime_like(
            history[spec.expires_time].dtype
        ):
            _assert_lossless_conversion(
                history[spec.expires_time],
                expiry_times,
                spec.expires_time,
                "expiry time",
            )
    # Time-zone alignment (localize/convert) never changes the instant, so it
    # needs no losslessness check; precision agreement is handled above.

    _assert_time_compatible(
        {
            "event time": event_times,
            "effective time": effective_times,
            "expiry time": expiry_times,
        },
        spec,
    )

    # -- event-time validity ------------------------------------------------
    event_time_missing = event_times.isna().to_numpy()
    evidence.n_invalid_event_time = int(event_time_missing.sum())

    # -- participant keys ---------------------------------------------------
    codes, _ = pd.factorize(
        pd.concat(
            [events[spec.participant_key], history[spec.participant_key]],
            ignore_index=True,
        ),
        use_na_sentinel=True,
    )
    event_codes = codes[: len(events)].astype(np.int64)
    history_codes = codes[len(events):].astype(np.int64)
    event_key_missing = events[spec.participant_key].isna().to_numpy()
    evidence.n_null_event_key = int(event_key_missing.sum())
    history_key_missing = history[spec.participant_key].isna().to_numpy()
    n_history_null_key = int(history_key_missing.sum())
    if n_history_null_key:
        evidence.warnings.append(
            f"{n_history_null_key} history row(s) have a null participant key and "
            "cannot be attached; they were ignored."
        )

    if effective_times.isna().any():
        bad = np.flatnonzero(effective_times.isna().to_numpy())
        raise CovariateJoinError(
            "History rows must declare an effective time; found null effective "
            f"time(s) at row(s) {_row_list(bad)}."
        )

    # -- declared expiry must be strictly after the effective time ----------
    effective_int = _time_to_int(effective_times)
    expiry_declared_int = _time_to_int(expiry_times)
    expiry_open = expiry_times.isna().to_numpy()
    declared_end_int = np.where(expiry_open, _OPEN_ENDED, expiry_declared_int).astype(
        np.int64
    )
    if spec.expires_time is not None:
        invalid_expiry = (~expiry_open) & (expiry_declared_int <= effective_int)
        if invalid_expiry.any():
            bad = np.flatnonzero(invalid_expiry)
            raise CovariateJoinError(
                "History rows must expire strictly after they take effect; "
                f"invalid interval(s) at row(s) {_row_list(bad)}."
            )

    # -- duplicate keys -----------------------------------------------------
    keep = np.ones(len(history), dtype=bool)
    keep &= ~history_key_missing

    candidate = np.flatnonzero(keep)
    if len(candidate):
        key_frame = pd.DataFrame(
            {
                "__key": history_codes[candidate],
                "__start": effective_int[candidate],
            }
        )
        sizes = key_frame.groupby(["__key", "__start"], sort=False).size()
        multi = sizes[sizes > 1]
        if len(multi):
            groups = key_frame.groupby(["__key", "__start"], sort=False).indices
            attributes_for_compare = history[list(sources)]
            for group_key in multi.index:
                positions = np.asarray(groups[group_key], dtype=np.int64)
                original_positions = candidate[positions]
                if _rows_identical(
                    attributes_for_compare, original_positions, expiry_times
                ):
                    keep[original_positions[1:]] = False
                    evidence.n_duplicate_rows_collapsed += len(original_positions) - 1
                    continue
                evidence.n_duplicate_conflicts += len(original_positions) - 1
                if spec.on_duplicate is OnDuplicate.ERROR:
                    raise CovariateJoinError(
                        "History rows duplicate the participant/effective-time key "
                        f"{group_key[0]!r} @ {effective_times.iloc[original_positions[0]]} "
                        f"with different attribute values at row(s) "
                        f"{_row_list(original_positions)}. Choose "
                        "on_duplicate='first' or 'last' to resolve this explicitly."
                    )
                if spec.on_duplicate is OnDuplicate.FIRST:
                    chosen = original_positions.min()
                else:
                    chosen = original_positions.max()
                dropped = original_positions[original_positions != chosen]
                keep[dropped] = False

    if spec.on_duplicate is OnDuplicate.ERROR and evidence.n_duplicate_rows_collapsed:
        evidence.warnings.append(
            f"{evidence.n_duplicate_rows_collapsed} identical duplicate history "
            "row(s) were collapsed."
        )

    # -- overlap resolution -------------------------------------------------
    active = np.flatnonzero(keep)
    order = np.lexsort((active, effective_int[active], history_codes[active]))
    active = active[order]
    sorted_key = history_codes[active]
    sorted_start = effective_int[active]
    sorted_declared_end = declared_end_int[active]

    group_ids = np.empty(len(active), dtype=np.int64)
    if len(active):
        boundaries = np.empty(len(active), dtype=bool)
        boundaries[0] = True
        boundaries[1:] = sorted_key[1:] != sorted_key[:-1]
        group_ids = np.cumsum(boundaries) - 1

    group_series = pd.Series(group_ids, dtype="int64")
    start_series = pd.Series(sorted_start, dtype="int64")
    # Shift within each participant using nullable Int64 so integer epoch
    # nanoseconds are never round-tripped through float.
    next_start = start_series.astype("Int64").groupby(group_series).shift(-1)
    has_next = next_start.notna().to_numpy()
    next_start_int = next_start.fillna(_OPEN_ENDED).to_numpy(dtype=np.int64)
    declared_end_series = pd.Series(sorted_declared_end, dtype="int64")

    if spec.on_overlap is OnOverlap.ERROR:
        overlaps = has_next & (sorted_declared_end > next_start_int)
        if overlaps.any():
            bad = active[np.flatnonzero(overlaps)]
            positions = np.flatnonzero(overlaps)
            detail = ", ".join(
                f"row {int(active[p])} expires "
                f"{expiry_times.iloc[int(active[p])]} after row "
                f"{int(active[p + 1]) if p + 1 < len(active) else '?'} takes effect"
                for p in positions[:3]
            )
            raise CovariateJoinError(
                "Overlapping versions of the same participant attribute: "
                f"{detail}. Declare on_overlap='latest_effective' or "
                "'earliest_effective' to resolve this explicitly."
            )
        start_adjusted_int = sorted_start.copy()
        end_adjusted_int = sorted_declared_end.copy()
    elif spec.on_overlap is OnOverlap.LATEST_EFFECTIVE:
        truncated = has_next & (sorted_declared_end > next_start_int)
        end_adjusted_int = np.minimum(sorted_declared_end, next_start_int)
        start_adjusted_int = sorted_start.copy()
        evidence.n_overlaps_resolved += int(truncated.sum())
    else:  # EARLIEST_EFFECTIVE
        previous_cursor = (
            declared_end_series.astype("Int64")
            .groupby(group_series)
            .cummax()
            .groupby(group_series)
            .shift(1)
        )
        previous_cursor = previous_cursor.fillna(_NEG_INF).to_numpy(dtype=np.int64)
        start_adjusted_int = np.maximum(sorted_start, previous_cursor)
        end_adjusted_int = sorted_declared_end.copy()
        evidence.n_overlaps_resolved += int((start_adjusted_int > sorted_start).sum())

    usable = start_adjusted_int < end_adjusted_int
    if spec.on_overlap is OnOverlap.EARLIEST_EFFECTIVE:
        shadowed = usable == False  # noqa: E712 - explicit for readability
        if shadowed.any():
            evidence.warnings.append(
                f"{int(shadowed.sum())} history row(s) never take effect because an "
                "earlier version is still valid; they cannot be attached."
            )
    usable_positions = active[usable]
    usable_start = start_adjusted_int[usable]
    usable_end = end_adjusted_int[usable]

    # -- pick one record per event -----------------------------------------
    # merge_asof only needs to rank candidate versions, so it runs on small
    # integer row ids.  All time comparisons stay in int64 to avoid the
    # float64 coercion merge_asof applies to large nanosecond values.
    n_events = len(events)
    chosen = np.full(n_events, -1, dtype=np.int64)
    if len(usable_positions):
        rank = np.arange(len(usable_positions), dtype=np.int64)
        left = pd.DataFrame(
            {
                "__t": _time_to_int(event_times),
                "__key": event_codes,
                "__pos": np.arange(n_events, dtype=np.int64),
            }
        )
        right = pd.DataFrame(
            {
                "__rank": rank,
                "__t": usable_start,
                "__key": history_codes[usable_positions],
            }
        )
        left_sorted = left.sort_values(["__t", "__key"], kind="mergesort")
        right_sorted = right.sort_values(["__t", "__key"], kind="mergesort")
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            on="__t",
            by="__key",
            direction="backward",
            allow_exact_matches=True,
        )
        candidate_rank = merged["__rank"].fillna(-1).to_numpy(dtype=np.int64)
        has_candidate = candidate_rank >= 0
        event_t_int = merged["__t"].to_numpy(dtype=np.int64)
        candidate_end = np.full(n_events, _NEG_INF, dtype=np.int64)
        candidate_orig = np.full(n_events, -1, dtype=np.int64)
        if has_candidate.any():
            candidate_end[has_candidate] = usable_end[candidate_rank[has_candidate]]
            candidate_orig[has_candidate] = usable_positions[candidate_rank[has_candidate]]
        still_valid = has_candidate & (candidate_end > event_t_int)
        chosen[merged["__pos"].to_numpy(dtype=np.int64)] = np.where(
            still_valid, candidate_orig, -1
        )

    # An event with a null participant key or an unparseable event time must
    # never match, even if the merge produced a candidate.
    chosen[event_key_missing] = -1
    chosen[event_time_missing] = -1

    # -- attach values and evidence ----------------------------------------
    attribute_frame = history[list(sources)].reset_index(drop=True)
    attribute_frame.columns = list(outputs)
    attached = pd.DataFrame(
        {
            output: _take_optional(attribute_frame[output], chosen)
            for output in outputs
        }
    )

    matched_mask = chosen >= 0
    null_value_mask = np.zeros(n_events, dtype=bool)
    if matched_mask.any():
        for output in outputs:
            values = attached[output].to_numpy(dtype=object, na_value=None)
            null_value_mask |= pd.isna(pd.Series(values)).to_numpy()
        null_value_mask &= matched_mask
    evidence.n_null_attribute_values = int(null_value_mask.sum())

    evidence.n_matched = int(matched_mask.sum())
    evidence.n_missing = int((~matched_mask).sum())
    evidence.n_history_rows_used = int(np.unique(chosen[matched_mask]).size)

    unusable_mask = (~matched_mask) | null_value_mask
    dropped_mask = np.zeros(n_events, dtype=bool)
    if unusable_mask.any():
        if spec.on_missing is OnMissing.ERROR:
            bad = np.flatnonzero(unusable_mask)
            null_keys = int(event_key_missing.sum())
            invalid_times = int(event_time_missing.sum())
            raise CovariateJoinError(
                f"{len(bad)} event(s) have no usable covariate record "
                f"({null_keys} null participant key(s), {invalid_times} invalid "
                f"event time(s), {int(null_value_mask.sum())} with null attribute "
                f"values). First affected event rows: {_row_list(events.index[bad])}. "
                "Choose on_missing='drop_event' or 'allow_na' to state the intended "
                "behaviour."
            )
        if spec.on_missing is OnMissing.DROP_EVENT:
            dropped_mask = unusable_mask
            evidence.n_dropped = int(dropped_mask.sum())

    evidence.n_history_rows_in = len(history)
    evidence.n_events_out = int(n_events - dropped_mask.sum())

    # -- assemble output (event rows keep their identity) -------------------
    status = np.where(
        matched_mask,
        np.where(null_value_mask, "null_attribute", "matched"),
        "missing",
    )

    # Per original history row: the effective window actually used (which can
    # differ from the declared one when overlap resolution truncated a
    # version).  ``_OPEN_ENDED`` becomes NaT (still in force / unbounded).
    start_by_original = _int_to_datetime(
        np.full(len(history), _OPEN_ENDED, dtype=np.int64),
        _OPEN_ENDED,
        effective_times.dtype,
    )
    end_by_original = start_by_original.copy()
    if len(usable_positions):
        start_by_original[usable_positions] = _int_to_datetime(
            usable_start, None, effective_times.dtype
        )
        end_by_original[usable_positions] = _int_to_datetime(
            usable_end, _OPEN_ENDED, effective_times.dtype
        )

    record_effective_used = _take_optional(
        pd.Series(start_by_original), chosen
    )
    record_expires = _take_optional(pd.Series(end_by_original), chosen)

    matched_rows = pd.array(chosen, dtype="Int64")
    matched_rows[chosen < 0] = pd.NA

    evidence_columns = pd.DataFrame(
        {
            f"{JOIN_EVIDENCE_PREFIX}matched_row": matched_rows,
            f"{JOIN_EVIDENCE_PREFIX}status": pd.Series(status, dtype="object"),
            f"{JOIN_EVIDENCE_PREFIX}record_effective": record_effective_used,
            f"{JOIN_EVIDENCE_PREFIX}record_expires": record_expires,
        }
    )

    out = pd.concat(
        [
            events.reset_index(drop=True),
            attached.reset_index(drop=True),
            evidence_columns.reset_index(drop=True),
        ],
        axis=1,
    )
    out.index = events.index.copy()
    out.index.name = events.index.name

    if dropped_mask.any():
        out = out.loc[~dropped_mask]
        status = status[~dropped_mask]
        matched_rows = matched_rows[~dropped_mask]
        record_effective_used = record_effective_used.loc[~dropped_mask].reset_index(
            drop=True
        )
        record_expires = record_expires.loc[~dropped_mask].reset_index(drop=True)
        event_times = event_times.loc[~dropped_mask].reset_index(drop=True)
        matched_mask = matched_mask[~dropped_mask]

    if len(out) > len(events):  # pragma: no cover - defensive invariant
        raise CovariateJoinError(
            "Internal error: the as-of join increased the event count."
        )

    evidence.matched_history_rows = matched_rows
    evidence.event_time = event_times.reset_index(drop=True)
    evidence.record_effective_time = record_effective_used
    evidence.record_expires_time = record_expires
    evidence.per_event_status = pd.Series(status, dtype="object")

    if spec.validate and variables is not None:
        evidence.validated_attributes = validate_attribute_schema(
            out, variables, outputs
        )

    evidence.source_fingerprint = fingerprint_frame(events)
    evidence.result_fingerprint = fingerprint_frame(out)

    return out, evidence


def _int_to_datetime(
    values: np.ndarray, open_sentinel: Optional[int], dtype
) -> np.ndarray:
    """Convert integer epoch values back to a datetime64/tz-aware numpy array.

    Values equal to ``open_sentinel`` (an exclusive upper bound) become NaT.
    """
    values = np.asarray(values, dtype=np.int64).copy()
    if open_sentinel is not None:
        values[values == open_sentinel] = np.iinfo(np.int64).min
    return pd.Series(values).astype("int64").astype(dtype).to_numpy()


def make_snapshot(
    data: pd.DataFrame,
    variables: Sequence[Any],
    *,
    is_joined: bool = False,
    evidence: Optional[JoinEvidence] = None,
) -> AnalysisSnapshot:
    variables = tuple(variables or ())
    return AnalysisSnapshot(
        data=data,
        variables=variables,
        schema=schema_from_variables(variables),
        fingerprint=fingerprint_frame(data),
        is_joined=is_joined,
        evidence=evidence,
    )
