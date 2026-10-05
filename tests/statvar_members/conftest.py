"""Shared fixtures and test doubles for the `statvar_members` test suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FakeSubjectPage:
    dcids: tuple[str, ...]
    truncated: bool


class FakeMemberArcReader:
    """Simulate reverse, forward, and name reads from an `arcs` mapping.

    Filters match by value membership, as in the live graph. Pages cap at `max_results` and
    `force_truncated` marks every page truncated. `node_arcs_many` ignores `individual_probe_budget`
    and resolves all dcids in one pass.
    """

    def __init__(
        self,
        *,
        arcs: Mapping[str, Mapping[str, tuple[str, ...]]],
        force_truncated: bool = False,
    ) -> None:
        self._arcs = arcs
        self._force_truncated = force_truncated
        self.calls: dict[str, int] = {
            "node_subjects": 0,
            "node_arcs_many": 0,
            "node_names_many": 0,
        }

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
    ) -> FakeSubjectPage:
        self.calls["node_subjects"] += 1
        matches = tuple(
            sorted(
                stat_var
                for stat_var, stat_var_arcs in self._arcs.items()
                if stat_var_arcs.get(property, ()) == (dcid,)
                and all(
                    value in stat_var_arcs.get(key, ()) for key, value in (filters or {}).items()
                )
            )
        )
        if self._force_truncated:
            return FakeSubjectPage(dcids=matches[:max_results], truncated=True)
        if len(matches) > max_results:
            return FakeSubjectPage(dcids=matches[:max_results], truncated=True)
        return FakeSubjectPage(dcids=matches, truncated=False)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self.calls["node_arcs_many"] += 1
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            stat_var_arcs = self._arcs.get(dcid)
            result[dcid] = (
                None
                if stat_var_arcs is None
                else {
                    prop: tuple(dict.fromkeys(stat_var_arcs.get(prop, ()))) for prop in properties
                }
            )
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return deduplicated `name` values followed by `alternateName` values.

        Missing dcids map to `()`.
        """
        self.calls["node_names_many"] += 1
        result: dict[str, tuple[str, ...]] = {}
        for dcid in dict.fromkeys(dcids):
            stat_var_arcs = self._arcs.get(dcid, {})
            result[dcid] = tuple(
                dict.fromkeys(
                    value
                    for prop in ("name", "alternateName")
                    for value in stat_var_arcs.get(prop, ())
                )
            )
        return result


class FlakyMemberArcReader(FakeMemberArcReader):
    """Simulate `/v2/node` omissions that occur on multi-dcid reads.

    `drop_first_batch` drops a property on the first multi-dcid appearance. `drop_second_batch`
    drops on the second, and `drop_always` drops on every request. Single-dcid reads return full
    arcs. The `empty_*` options return `None` for the entire arc map. Their dcids remain in `arcs`
    to model a reverse-index hit with an empty forward read.
    """

    def __init__(
        self,
        *,
        arcs: Mapping[str, Mapping[str, tuple[str, ...]]],
        drop_first_batch: Mapping[str, str] | None = None,
        drop_second_batch: Mapping[str, str] | None = None,
        drop_always: Mapping[str, str] | None = None,
        empty_first_batch: Sequence[str] | None = None,
        empty_always: Sequence[str] | None = None,
    ) -> None:
        super().__init__(arcs=arcs)
        self._drop_first_batch = dict(drop_first_batch or {})
        self._drop_second_batch = dict(drop_second_batch or {})
        self._drop_always = dict(drop_always or {})
        self._empty_first_batch = set(empty_first_batch or ())
        self._empty_always = set(empty_always or ())
        self._already_dropped: set[str] = set()
        self._already_dropped_second: set[str] = set()
        self._multi_dcid_appearances: dict[str, int] = {}
        self._already_emptied: set[str] = set()

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self.calls["node_arcs_many"] += 1
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            stat_var_arcs = self._arcs.get(dcid)
            if stat_var_arcs is None:
                result[dcid] = None
                continue
            if dcid in self._empty_always:
                result[dcid] = None
                continue
            if (
                dcid in self._empty_first_batch
                and dcid not in self._already_emptied
                and len(dcids) > 1
            ):
                self._already_emptied.add(dcid)
                result[dcid] = None
                continue
            if len(dcids) > 1:
                self._multi_dcid_appearances[dcid] = self._multi_dcid_appearances.get(dcid, 0) + 1
            arcs = {prop: tuple(dict.fromkeys(stat_var_arcs.get(prop, ()))) for prop in properties}
            always_drop = self._drop_always.get(dcid)
            if always_drop in arcs:
                arcs[always_drop] = ()
            first_drop = self._drop_first_batch.get(dcid)
            if first_drop in arcs and dcid not in self._already_dropped and len(dcids) > 1:
                arcs[first_drop] = ()
                self._already_dropped.add(dcid)
            second_drop = self._drop_second_batch.get(dcid)
            if (
                second_drop in arcs
                and dcid not in self._already_dropped_second
                and len(dcids) > 1
                and self._multi_dcid_appearances.get(dcid, 0) >= 2
            ):
                arcs[second_drop] = ()
                self._already_dropped_second.add(dcid)
            result[dcid] = arcs
        return result


_KENYA_LIFE_EXPECTANCY_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "LifeExpectancy_Person": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
    },
    "LifeExpectancy_Person_Male": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Male",),
    },
    "LifeExpectancy_Person_Female": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Female",),
    },
    "LifeExpectancy_Person_0Years": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("age",),
        "observationProperties": (),
        "age": ("Years0",),
    },
    "LifeExpectancy_Person_0Years_Male": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("age", "gender"),
        "observationProperties": (),
        "age": ("Years0",),
        "gender": ("Male",),
    },
}
"""A slice of the 351-member dev-verified family, covering antichain reduction and
undeclared age slots."""

_MALE_FEMALE_ONLY_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "Count_Person_Male": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Male",),
    },
    "Count_Person_Female": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Female",),
    },
}
"""The `Person/count/measuredValue` family has two gendered members and no subset relation."""


def kenya_reader(*, force_truncated: bool = False) -> FakeMemberArcReader:
    return FakeMemberArcReader(arcs=_KENYA_LIFE_EXPECTANCY_ARCS, force_truncated=force_truncated)


def male_female_only_reader(*, force_truncated: bool = False) -> FakeMemberArcReader:
    return FakeMemberArcReader(arcs=_MALE_FEMALE_ONLY_ARCS, force_truncated=force_truncated)
