"""Tests graph confirmation's entity probes, coverage labels, and binding filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from search.engine.bind import SlotBindingDraft
from search.engine.extract import DateRequest
from search.engine.retrieval.retrieve import Materialised, NoDataDraft
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.retrieval.standard import standard_resolve
from search.engine.shape import ShapeDraft
from search.models import TimeWindow
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._discover_fixtures import (
    MEAS_PROP_DCID,
    POP_TYPE_DCID,
    PROP_PURPOSE,
    PROP_RECIPIENT,
    PROP_SCHEME,
    STAT_TYPE_DCID,
    _three_constraint_arcs,
)
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import dimensions_of
from tests.fixtures import FakeGraph

if TYPE_CHECKING:
    from series_coverage.shared import CoverageSnapshot


def _dated(years: range) -> dict:
    dates = [str(y) for y in years]
    return {
        "facetId": "f",
        "provenanceId": "dc/base/Source",
        "earliestDate": dates[0],
        "latestDate": dates[-1],
        "obsCount": len(dates),
        "dates": dates,
        "values": [1.0] * len(dates),
    }


def test_a_standard_plan_reads_the_members_holding_a_point_in_its_years() -> None:
    """The unconstrained member holds data at Kenya only in 2000 to 2004 and its Health part
    for 2018 to 2020. Its arcs are unread, so its constraints are unknown and it dominates
    no member. A 2015 to 2020 read plans the part, the one member with a point there; a
    dateless read also plans the part, because a member whose constraints are known stands
    in ahead of one whose constraints are not. A nearest-real relaxation of a standard shape
    resolves through the same `standard_resolve`."""
    total, health = "sv/WindowTotal", "sv/WindowHealth"
    shape = ShapeDraft(
        shape_id="test_window_antichain",
        label="test shape",
        pop_type_dcid="TestWindowPopulation",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            total: {},
            health: {
                "constraintProperties": {"nodes": [{"dcid": "sector"}]},
                "sector": {"nodes": [{"dcid": "Health"}]},
            },
        },
    )
    graph = FakeGraph(
        nodes={},
        obs={
            f"{total}|country/KEN": [_dated(range(2000, 2005))],
            f"{health}|country/KEN": [_dated(range(2018, 2021))],
        },
        detect={},
        resolve={},
    )

    def plan_members(date_request: DateRequest | None) -> tuple[str, ...]:
        result = standard_resolve(
            shape=shape,
            bindings=[],
            entities={"observationAbout": ["country/KEN"]},
            graph=graph,
            date_request=date_request,
        )
        assert isinstance(result, Materialised) and result.has_data
        assert result.read_plan is not None
        return result.read_plan.members

    window = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    assert plan_members(window) == (health,)
    assert plan_members(None) == (health,)


# A debtor-only flow: the role graph confirmation reads coverage for.
_DEBTOR = ("debtor",)


def _debtor_arcs(arcs: dict) -> dict:
    """`arcs` declaring the debtor role."""
    return {**arcs, "observationProperties": {"nodes": [{"dcid": "debtor"}]}}


def _debtor_coverage(sv_dcid: str, recipient: str) -> CoverageSnapshot:
    """Coverage holding `sv_dcid` at `recipient` for 2010 to 2020."""
    index = build_coverage_index(
        [(sv_dcid, "f1", ABSENT_ENTITY, recipient, list(range(2010, 2021)))],
        role_names={sv_dcid: RoleNames(donor=None, recipient="debtor")},
        facet_metadata={"f1": FacetMetadata(provenance_id="dc/base/Source")},
    )
    return FakeCoverageService(index).snapshot()


# ---------------------------------------------------------------------------
# graph_confirm_resolve: coverage dimension labels
# ---------------------------------------------------------------------------


class TestGraphConfirmResolveLabels:
    """Custom labels replace the default ``sources`` and ``observations`` dimensions."""

    def _shape_with_sv(self, sv_dcid: str) -> ShapeDraft:
        return ShapeDraft(
            shape_id="test_shape_with_custom_labels",
            label="test shape",
            pop_type_dcid=POP_TYPE_DCID,
            meas_prop_dcid=MEAS_PROP_DCID,
            stat_type_dcid=STAT_TYPE_DCID,
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            sv_arc_facts={
                sv_dcid: _debtor_arcs(
                    _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
                )
            },
            declared_roles=_DEBTOR,
        )

    def test_custom_labels_used_when_given(self):
        sv = "sv/SomeHealthSV"
        result = graph_confirm_resolve(
            shape=self._shape_with_sv(sv),
            bindings=[],
            subject_dcids=["country/ETH"],
            origin_dcids=None,
            graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
            facet_label="donors",
            obs_label="years",
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage_snapshot=_debtor_coverage(sv, "country/ETH"),
        )
        assert isinstance(result, Materialised)
        # The variables dimension counts resolved members.
        # Other dimensions count facets or observations.
        assert {d.label for d in dimensions_of(result.coverage)} == {"donors", "years", "variables"}

    def test_default_labels_are_sources_observations(self):
        sv = "sv/SomeHealthSV"
        result = graph_confirm_resolve(
            shape=self._shape_with_sv(sv),
            bindings=[],
            subject_dcids=["country/ETH"],
            origin_dcids=None,
            graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage_snapshot=_debtor_coverage(sv, "country/ETH"),
        )
        assert isinstance(result, Materialised)
        # "variables" is a third, always-appended dimension -- see the
        # custom-labels test above.
        assert {d.label for d in dimensions_of(result.coverage)} == {
            "sources",
            "observations",
            "variables",
        }


# ---------------------------------------------------------------------------
# graph_confirm_resolve: unresolved bound-constraint values fail closed
# ---------------------------------------------------------------------------

_UNRESOLVED_PURPOSE_NODE = {
    "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
}


class TestGraphConfirmResolveUnresolvedConstraint:
    """A member whose constraints do not all resolve is one the mint's walk does not admit
    (`statvar_members.constraints_resolve`), and Search lists none. A bound purpose that
    does not resolve refuses as ``binding_mismatch``, an unbound one as
    ``variable_not_resolved``.

    The shape is subject-only, and the rule applies to every shape. Fixtures keep the
    schema tuple and recipient fixed."""

    _RECIPIENT = "country/ETH"

    def _arcs(self, *, purpose: dict | None) -> dict:
        constraint_nodes = [{"dcid": PROP_SCHEME}, {"dcid": PROP_RECIPIENT}]
        arcs: dict = {
            "populationType": {"nodes": [{"dcid": POP_TYPE_DCID}]},
            "measuredProperty": {"nodes": [{"dcid": MEAS_PROP_DCID}]},
            "statType": {"nodes": [{"dcid": STAT_TYPE_DCID}]},
            PROP_SCHEME: {"nodes": [{"dcid": "ODAGrants"}]},
            PROP_RECIPIENT: {"nodes": [{"dcid": self._RECIPIENT}]},
        }
        if purpose is not None:
            constraint_nodes.append({"dcid": PROP_PURPOSE})
            arcs[PROP_PURPOSE] = purpose
        arcs["constraintProperties"] = {"nodes": constraint_nodes}
        return _debtor_arcs(arcs)

    def _shape(self, sv_dcid: str, arcs: dict) -> ShapeDraft:
        return ShapeDraft(
            shape_id="test_confirm_unresolved_shape",
            label="test shape",
            pop_type_dcid=POP_TYPE_DCID,
            meas_prop_dcid=MEAS_PROP_DCID,
            stat_type_dcid=STAT_TYPE_DCID,
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            sv_arc_facts={sv_dcid: arcs},
            declared_roles=_DEBTOR,
        )

    def _resolve(self, sv_dcid: str, arcs: dict, bindings: list[SlotBindingDraft]):
        return graph_confirm_resolve(
            shape=self._shape(sv_dcid, arcs),
            bindings=bindings,
            subject_dcids=[self._RECIPIENT],
            origin_dcids=None,
            graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage_snapshot=_debtor_coverage(sv_dcid, self._RECIPIENT),
        )

    def test_unresolved_bound_property_excludes_sv(self):
        sv = "Test_Confirm_UnresolvedPurpose"
        arcs = self._arcs(purpose=_UNRESOLVED_PURPOSE_NODE)
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid=PROP_PURPOSE, kind="value", value_dcids=["DAC/Health"]
            )
        ]
        result = self._resolve(sv, arcs, bindings)
        assert isinstance(result, NoDataDraft)
        assert result.reason == "binding_mismatch", (
            "the filter emptied a real candidate set on a present-but-unresolved "
            "bound property; variable_not_resolved would wrongly claim no StatVar "
            f"matched at all, got {result.reason!r}"
        )

    def test_resolved_matching_bound_property_still_included(self):
        sv = "Test_Confirm_ResolvedMatch"
        arcs = self._arcs(purpose={"nodes": [{"dcid": "DAC/Health"}]})
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid=PROP_PURPOSE, kind="value", value_dcids=["DAC/Health"]
            )
        ]
        result = self._resolve(sv, arcs, bindings)
        assert isinstance(result, Materialised)
        assert result.sv_dcids == [sv]

    def test_resolved_mismatched_bound_property_still_excluded(self):
        sv = "Test_Confirm_ResolvedMismatch"
        arcs = self._arcs(purpose={"nodes": [{"dcid": "DAC/Water"}]})
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid=PROP_PURPOSE, kind="value", value_dcids=["DAC/Health"]
            )
        ]
        result = self._resolve(sv, arcs, bindings)
        assert isinstance(result, NoDataDraft)
        assert result.reason == "binding_mismatch", (
            "the filter emptied a real candidate set on a resolved-but-mismatched "
            f"bound property, got {result.reason!r}"
        )

    def test_absent_bound_property_excludes_sv(self):
        """A member silent on a bound property is not served under it, as on the
        index path and at the mint and the resubmit."""
        sv = "Test_Confirm_AbsentPurpose"
        arcs = self._arcs(purpose=None)
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid=PROP_PURPOSE, kind="value", value_dcids=["DAC/Health"]
            )
        ]
        result = self._resolve(sv, arcs, bindings)
        assert isinstance(result, NoDataDraft)
        assert result.reason == "binding_mismatch"

    def test_unbound_property_unresolved_excludes_sv_as_the_mint_does(self):
        """The mint's walk admits no member asserting an unresolved constraint, bound or not
        (`statvar_members.constraints_resolve`), so Search lists none. With nothing bound,
        no member is left to mismatch."""
        sv = "Test_Confirm_UnresolvedUnbound"
        arcs = self._arcs(purpose=_UNRESOLVED_PURPOSE_NODE)
        result = self._resolve(sv, arcs, bindings=[])
        assert isinstance(result, NoDataDraft)
        assert result.reason == "variable_not_resolved"

    def test_matching_bound_value_beside_an_unresolved_unbound_property_is_not_a_mismatch(self):
        """The member asserts the bound scheme, so no bound value is mismatched. Its unbound
        purpose does not resolve, so the mint's walk does not admit it, and the answer
        refuses as for no member."""
        sv = "Test_Confirm_MatchingAndUnresolvedUnbound"
        arcs = self._arcs(purpose=_UNRESOLVED_PURPOSE_NODE)
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid=PROP_SCHEME, kind="value", value_dcids=["ODAGrants"]
            )
        ]
        result = self._resolve(sv, arcs, bindings)
        assert isinstance(result, NoDataDraft)
        assert result.reason == "variable_not_resolved"

    def test_a_declared_property_with_no_value_arc_excludes_sv_as_the_mint_does(self):
        """The member lists purpose in constraintProperties but carries no purpose arc. The
        mint's walk rejects it for the absent value (`statvar_members.constraints_resolve`),
        so Search lists it under no binding."""
        sv = "Test_Confirm_DeclaredWithoutValue"
        arcs = self._arcs(purpose=None)
        arcs["constraintProperties"]["nodes"].append({"dcid": PROP_PURPOSE})
        result = self._resolve(sv, arcs, bindings=[])
        assert isinstance(result, NoDataDraft)
        assert result.reason == "variable_not_resolved"

    def test_matching_and_unresolved_bound_properties_excludes_sv(self):
        """A matching scheme cannot hide an unresolved bound purpose."""
        sv = "Test_Confirm_MatchingAndUnresolved"
        arcs = self._arcs(purpose=_UNRESOLVED_PURPOSE_NODE)
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid=PROP_SCHEME, kind="value", value_dcids=["ODAGrants"]
            ),
            SlotBindingDraft(
                axis="how", property_dcid=PROP_PURPOSE, kind="value", value_dcids=["DAC/Health"]
            ),
        ]
        result = self._resolve(sv, arcs, bindings)
        assert isinstance(result, NoDataDraft)
        assert result.reason == "binding_mismatch", (
            "a resolved-and-matching property alongside a present-but-unresolved "
            f"one must still empty the filter as binding_mismatch, got {result.reason!r}"
        )

    def test_no_recipient_still_reports_variable_not_resolved(self):
        """No recipient returns ``variable_not_resolved`` before filtering."""
        result = graph_confirm_resolve(
            shape=self._shape("Test_Confirm_NoRecipient", self._arcs(purpose=None)),
            bindings=[],
            subject_dcids=[],
            origin_dcids=None,
            graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage_snapshot=empty_coverage_snapshot(),
        )
        assert isinstance(result, NoDataDraft)
        assert result.reason == "variable_not_resolved"

    def test_no_arc_facts_still_reports_variable_not_resolved(self):
        """An empty candidate set returns ``variable_not_resolved`` before filtering."""
        empty_shape = ShapeDraft(
            shape_id="test_confirm_no_candidates_shape",
            label="test shape",
            pop_type_dcid=POP_TYPE_DCID,
            meas_prop_dcid=MEAS_PROP_DCID,
            stat_type_dcid=STAT_TYPE_DCID,
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            sv_arc_facts={},
            declared_roles=_DEBTOR,
        )
        result = graph_confirm_resolve(
            shape=empty_shape,
            bindings=[],
            subject_dcids=[self._RECIPIENT],
            origin_dcids=None,
            graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage_snapshot=empty_coverage_snapshot(),
        )
        assert isinstance(result, NoDataDraft)
        assert result.reason == "variable_not_resolved"
