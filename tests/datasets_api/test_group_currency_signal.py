"""Pins currency signals for combined requests to the facets dispatched by every member.
The shared selection may differ from each member's individually ranked top facet.
Tests cover member-only changes, shared-facet date changes, and incoherent selections.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import httpx

from datasets_api.currency import derive_signals
from datasets_api.versions import MemberSignal
from materialise import MaterialisationRequest
from materialise.facets import (
    CollectionFacetProbe,
    CoverageAdmission,
    FacetBinding,
    MemberProbe,
    UnitRequest,
)
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.window import RequestedWindow
from series_coverage.schema import CoverageManifest, FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import build_coverage_index

from ._graph_doubles import FakeMintGraphReader
from .test_definition_door_currency import _client, _FakeHeadStore

_GRANTS = "ODAGrantsAmount"
_GRANTS_REF = f"dcid:{_GRANTS}"
_LOANS = "ODALoansAmount"
_LOANS_REF = f"dcid:{_LOANS}"
_RECIPIENT = "country/KEN"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _coverage_probe(candidates: tuple[tuple[Facet, ...], ...]) -> CollectionFacetProbe:
    """Every request here is role-bearing, so its candidates come from coverage."""
    return CollectionFacetProbe(
        members=tuple(
            MemberProbe(candidates=member, admission=CoverageAdmission()) for member in candidates
        )
    )


_MEMBER_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "populationType": ("DevelopmentFinance",),
    "measuredProperty": ("amount",),
    "observationProperties": ("recipient",),
    "constraintProperties": ("DevelopmentFinanceScheme",),
}

_SHARED = "shared"
_A_ONLY = "aonly"
_B_ONLY = "bonly"

_IDENTITY = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [
        {
            "property": "DevelopmentFinanceScheme",
            "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
        }
    ],
    "entities": {"recipient": [_RECIPIENT]},
    "requested_window": {"kind": "all"},
}


def _graph() -> FakeMintGraphReader:
    """Role-bearing group probes read coverage, so this graph needs no facet map."""
    return FakeMintGraphReader(
        arcs={
            _GRANTS: {**_MEMBER_ARCS, "DevelopmentFinanceScheme": ("ODAGrants",)},
            _LOANS: {**_MEMBER_ARCS, "DevelopmentFinanceScheme": ("ODALoans",)},
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "WorldBank": "World Bank", _RECIPIENT: "Kenya"},
    )


def _row(
    *,
    ref: str,
    value: str,
    period: str = "2020",
    facet_id: str = _SHARED,
    measurement_method: str = "GrantEquivalent",
) -> str:
    """Builds an SDMX row with a bare StatVar id. `ref` uses the `dcid:` form.
    Coverage ranks candidates. This row supplies period and count for the selected fetch.
    """
    bare_stat_var = ref.removeprefix("dcid:")
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare_stat_var},{_RECIPIENT},USD,{measurement_method},"
        f"NotApplicable,OECD_DAC2A,{period},{value},,{facet_id}\r\n"
    )


def _sdmx_transport(rows_by_ref: dict[str, dict[str, str]]):
    """Routes each member's fetch by bare StatVar id.
    Facet probes use coverage.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        variable_param = request.url.params.get("c[variableMeasured]", "")
        rows = "".join(
            rows_by_ref[ref]["text"]
            for stat_var, ref in ((_GRANTS, _GRANTS_REF), (_LOANS, _LOANS_REF))
            if stat_var in variable_param
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _spec_id_and_sidecar() -> tuple[str, dict]:
    recomputed = recompute_identity(_IDENTITY)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": _IDENTITY,
    }


def _group_coverage_rows() -> dict[tuple[str, str], dict[str, object]]:
    """Each member has a shared one-observation facet and a two-observation member-only facet."""
    return {
        (_GRANTS, _SHARED): {"measurement_method": "GrantEquivalent", "years": [2020]},
        (_GRANTS, _A_ONLY): {"measurement_method": "Disbursement", "years": [2020, 2021]},
        (_LOANS, _SHARED): {"measurement_method": "GrantEquivalent", "years": [2020]},
        (_LOANS, _B_ONLY): {"measurement_method": "Disbursement", "years": [2020, 2021]},
    }


class _MutableCoverageService:
    """Rebuilds coverage from mutable rows so tests can change the next probe's result."""

    def __init__(
        self, rows: dict[tuple[str, str], dict[str, object]], *, stamp: str = "test-coverage-stamp"
    ) -> None:
        self._rows = rows
        self.stamp = stamp

    def current(self):
        return build_coverage_index(
            [
                (stat_var, facet_id, "country/USA", _RECIPIENT, spec["years"])
                for (stat_var, facet_id), spec in self._rows.items()
            ],
            role_names={
                _GRANTS: RoleNames(donor="provider", recipient="recipient"),
                _LOANS: RoleNames(donor="provider", recipient="recipient"),
            },
            facet_metadata={
                facet_id: FacetMetadata(
                    provenance_id="OECD_DAC2A",
                    measurement_method=spec["measurement_method"],
                    unit="USD",
                )
                for (_stat_var, facet_id), spec in self._rows.items()
            },
        )

    def snapshot(self) -> CoverageSnapshot:
        return CoverageSnapshot(
            index=self.current(),
            manifest=CoverageManifest(
                stamp=self.stamp,
                build_id=f"{self.stamp}-build",
                built_at="t",
                source="fake",
                row_counts={},
                files={},
                horizon=self.current().horizon,
            ),
            stamp=self.stamp,
        )


def test_group_signal_tracks_the_shared_facet_not_either_members_own_top():
    rows_by_ref = {
        _GRANTS_REF: {"text": _row(ref=_GRANTS_REF, value="1000")},
        _LOANS_REF: {"text": _row(ref=_LOANS_REF, value="500")},
    }
    coverage_rows = _group_coverage_rows()
    coverage = _MutableCoverageService(coverage_rows)
    transport = _sdmx_transport(rows_by_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=_graph(), sdmx_transport=transport, head_writer=head_store, coverage=coverage
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text

    # Each member has a shared facet (one observation) and a higher-ranked member-only facet.
    # The group selects the shared facet and sums its counts. Coverage gives year-grain dates.
    assert head_store.head_for(spec_id).signal == (MemberSignal(_SHARED, 2, "2020", "2020"),)

    # The unchanged head serves this request after a member-only facet change.
    head_store.age_head(spec_id, age=timedelta(minutes=20))
    coverage_rows[(_GRANTS, _A_ONLY)] = {
        "measurement_method": "Disbursement",
        "years": [2020, 2021, 2030],
    }
    unaffected = client.get(f"/api/datasets/{spec_id}")
    assert unaffected.status_code == 200, unaffected.text
    assert unaffected.json()["trace"]["served_from"] == "head"
    assert head_store.head_for(spec_id).content_version == minted.json()["content_version"]

    # Moving member A's shared-facet year to 2021 widens the signal and forces rematerialisation.
    head_store.age_head(spec_id, age=timedelta(minutes=20))
    coverage_rows[(_GRANTS, _SHARED)] = {"measurement_method": "GrantEquivalent", "years": [2021]}
    rows_by_ref[_GRANTS_REF]["text"] = _row(ref=_GRANTS_REF, value="2000", period="2021")
    changed = client.get(f"/api/datasets/{spec_id}")
    assert changed.status_code == 200, changed.text
    assert changed.json()["trace"]["served_from"] == "materialise"
    assert head_store.head_for(spec_id).content_version == changed.json()["content_version"]
    assert head_store.head_for(spec_id).signal == (MemberSignal(_SHARED, 2, "2020", "2021"),)


_NO_DISPATCH = MemberSignal(None, 0, None, None)


def test_an_empty_probe_mints_the_empty_result_and_its_head_serves_until_data_arrives():
    rows_by_ref = {
        _GRANTS_REF: {"text": _row(ref=_GRANTS_REF, value="1000")},
        _LOANS_REF: {"text": _row(ref=_LOANS_REF, value="500")},
    }
    # Coverage holds no series for either member, so no probe returns a candidate.
    coverage_rows: dict[tuple[str, str], dict[str, object]] = {}
    head_store = _FakeHeadStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(rows_by_ref),
        head_writer=head_store,
        coverage=_MutableCoverageService(coverage_rows),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
            "options": {"bundle": False},
        },
    )

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    assert manifest["coverage"]["rows"] == 0
    assert "empty_result" in {warning["code"] for warning in manifest["warnings"]}
    assert manifest["citation"]["sources"] == []
    assert head_store.head_for(spec_id).signal == (_NO_DISPATCH,)

    # The probe is still empty, so the head serves the stored empty version.
    head_store.age_head(spec_id, age=timedelta(minutes=20))
    unchanged = client.get(f"/api/datasets/{spec_id}")
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json()["trace"]["served_from"] == "head"
    assert unchanged.json()["content_version"] == manifest["content_version"]

    # A rendering the empty version does not hold yet is built from its stored frame.
    parquet = client.get(f"/api/datasets/{spec_id}?format=parquet")
    assert parquet.status_code == 200, parquet.text
    assert parquet.json()["content_version"] == manifest["content_version"]

    # Data arrives, the signal moves, and the definition rematerialises with rows.
    head_store.age_head(spec_id, age=timedelta(minutes=20))
    coverage_rows.update(_group_coverage_rows())
    arrived = client.get(f"/api/datasets/{spec_id}")
    assert arrived.status_code == 200, arrived.text
    assert arrived.json()["trace"]["served_from"] == "materialise"
    assert arrived.json()["coverage"]["rows"] == 2


def test_a_single_member_whose_probe_is_empty_signals_no_dispatched_facet():
    request = MaterialisationRequest(
        stat_var=_GRANTS_REF,
        role_bindings={"recipient": [_RECIPIENT]},
        declared_roles=("recipient",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )

    signals = derive_signals([request], _coverage_probe(((),)), graph=FakeMintGraphReader())

    assert signals == (_NO_DISPATCH,)


def test_group_signal_is_none_when_the_shared_facet_carries_no_provenance():
    # Without provenance or a declared default, real dispatch raises `DefaultFacetUndefinedError`.
    # The currency check must therefore return no signal.
    shared_no_provenance_a = Facet(
        facet_id=_SHARED,
        provenance_id=None,
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    shared_no_provenance_b = Facet(
        facet_id=_SHARED,
        provenance_id=None,
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = [
        MaterialisationRequest(
            stat_var=_GRANTS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
        MaterialisationRequest(
            stat_var=_LOANS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
    ]
    probe = _coverage_probe(((shared_no_provenance_a,), (shared_no_provenance_b,)))
    graph = FakeMintGraphReader()

    assert derive_signals(requests, probe, graph=graph) is None


def test_member_signal_is_none_when_a_request_form_unit_resolves_to_a_non_currency_native_unit():
    # A request-form conversion requires a currency-native unit. Plain `USD` fails that check, so
    # signal derivation returns `None`, matching the refusal a rematerialisation would raise.
    candidate = Facet(
        facet_id="facet1",
        provenance_id="PROV1",
        measurement_method="METH1",
        unit="USD",  # not "CurrencyUSD" -- fails the currency-native check
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )
    request = MaterialisationRequest(
        stat_var="dcid:Exemplar",
        role_bindings={"recipient": [_RECIPIENT]},
        declared_roles=("recipient",),
        facet_bindings={
            "unit": FacetBinding(
                state="request", request=UnitRequest(currency="USD", prices="current")
            )
        },
        window=RequestedWindow(kind="all"),
    )
    probe = _coverage_probe(((candidate,),))
    graph = FakeMintGraphReader()

    assert derive_signals([request], probe, graph=graph) is None


def test_group_signal_refuses_when_declared_policies_name_no_jointly_carried_combination():
    # The provenance policy selects `f1` and the unit policy selects `f2`, producing no coherent
    # candidate. Dispatch refuses this combination, so signal derivation must return `None`.
    facet_f1 = Facet(
        facet_id="f1",
        provenance_id="P1",
        measurement_method="M1",
        unit="CurrencyUSD",
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )
    facet_f2 = Facet(
        facet_id="f2",
        provenance_id="P2",
        measurement_method="M2",
        unit="CurrencyEUR",
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )
    provenance_policy = FacetDefaultPolicy(
        resolved_ref="P1",
        policy_node="policy/provenance-default",
        policy_version="v1",
        selection_reason="declared",
    )
    unit_policy = FacetDefaultPolicy(
        resolved_ref="CurrencyEUR",
        policy_node="policy/unit-default",
        policy_version="v1",
        selection_reason="declared",
    )
    requests = [
        MaterialisationRequest(
            stat_var=_GRANTS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
        MaterialisationRequest(
            stat_var=_LOANS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
    ]
    probe = _coverage_probe(((facet_f1, facet_f2), (facet_f1, facet_f2)))
    graph = FakeMintGraphReader(
        policies={
            (_GRANTS_REF, "provenance"): provenance_policy,
            (_LOANS_REF, "provenance"): provenance_policy,
            (_GRANTS_REF, "unit"): unit_policy,
            (_LOANS_REF, "unit"): unit_policy,
        }
    )

    assert derive_signals(requests, probe, graph=graph) is None


def test_group_signal_sums_every_dispatched_facet_in_an_index_sourced_group_not_just_the_top_one():
    # Index-sourced members agree on provenance and unit, then dispatch their own facets. Their
    # counts (5 + 3) form the group total, even though `facetGrants` ranks first.
    facet_grants = Facet(
        facet_id="facetGrants",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    facet_loans = Facet(
        facet_id="facetLoans",
        provenance_id="OECD_DAC2A",
        measurement_method="Disbursement",
        unit="USD",
        obs_count=3,
        earliest_date="2019",
        latest_date="2019",
    )
    requests = [
        MaterialisationRequest(
            stat_var=_GRANTS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
            index_sourced=True,
        ),
        MaterialisationRequest(
            stat_var=_LOANS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
            index_sourced=True,
        ),
    ]
    probe = _coverage_probe(((facet_grants,), (facet_loans,)))
    graph = FakeMintGraphReader()

    signals = derive_signals(requests, probe, graph=graph)

    assert signals is not None
    (signal,) = signals
    # `"facetGrants"` sorts before `"facetLoans"`, the lowest dispatched
    # facet_id, naming the merged signal.
    assert signal.dispatch_facet_id == "facetGrants"
    assert signal.obs_count == 8
    assert signal.earliest_date == "2019"
    assert signal.latest_date == "2020"


def test_group_signal_widens_when_a_row_moves_on_the_non_top_dispatched_facet_of_an_index_sourced_group():
    # Equal counts let the group select `facetGrants` by date, but it dispatches both facets, so
    # moving `facetLoans` to 2022 widens the merged signal.
    facet_grants = Facet(
        facet_id="facetGrants",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=3,
        earliest_date="2019",
        latest_date="2020",
    )
    facet_loans = Facet(
        facet_id="facetLoans",
        provenance_id="OECD_DAC2A",
        measurement_method="Disbursement",
        unit="USD",
        obs_count=3,
        earliest_date="2019",
        latest_date="2019",
    )
    requests = [
        MaterialisationRequest(
            stat_var=_GRANTS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
            index_sourced=True,
        ),
        MaterialisationRequest(
            stat_var=_LOANS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
            index_sourced=True,
        ),
    ]
    graph = FakeMintGraphReader()
    probe = _coverage_probe(((facet_grants,), (facet_loans,)))

    (before,) = derive_signals(requests, probe, graph=graph)
    assert before.dispatch_facet_id == "facetGrants"
    assert before.latest_date == "2020"

    moved_loans = replace(facet_loans, latest_date="2022")
    moved_probe = _coverage_probe(((facet_grants,), (moved_loans,)))
    (after,) = derive_signals(requests, moved_probe, graph=graph)

    assert after.latest_date == "2022"
    assert after != before


def test_group_signal_is_unaffected_by_a_facet_outside_the_agreement_key():
    # `facetOther` has a different provenance, so it cannot join the shared selection. Changing
    # its date must leave the signal unchanged.
    facet_grants = Facet(
        facet_id="facetGrants",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=3,
        earliest_date="2019",
        latest_date="2020",
    )
    facet_loans = Facet(
        facet_id="facetLoans",
        provenance_id="OECD_DAC2A",
        measurement_method="Disbursement",
        unit="USD",
        obs_count=3,
        earliest_date="2019",
        latest_date="2019",
    )
    facet_other_provenance = Facet(
        facet_id="facetOther",
        provenance_id="OTHER_PROV",
        measurement_method="Disbursement",
        unit="USD",
        obs_count=3,
        earliest_date="2019",
        latest_date="2019",
    )
    requests = [
        MaterialisationRequest(
            stat_var=_GRANTS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
            index_sourced=True,
        ),
        MaterialisationRequest(
            stat_var=_LOANS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
            index_sourced=True,
        ),
    ]
    graph = FakeMintGraphReader()
    probe = _coverage_probe(((facet_grants,), (facet_loans, facet_other_provenance)))

    (before,) = derive_signals(requests, probe, graph=graph)

    moved_other = replace(facet_other_provenance, latest_date="2099")
    moved_probe = _coverage_probe(((facet_grants,), (facet_loans, moved_other)))
    (after,) = derive_signals(requests, moved_probe, graph=graph)

    assert after == before


def test_group_signal_refuses_the_same_group_when_no_member_is_index_sourced():
    # Without `index_sourced=True`, members must share a `facet_id`, and these candidates differ.
    facet_grants = Facet(
        facet_id="facetGrants",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    facet_loans = Facet(
        facet_id="facetLoans",
        provenance_id="OECD_DAC2A",
        measurement_method="Disbursement",
        unit="USD",
        obs_count=3,
        earliest_date="2020",
        latest_date="2020",
    )
    requests = [
        MaterialisationRequest(
            stat_var=_GRANTS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
        MaterialisationRequest(
            stat_var=_LOANS_REF,
            role_bindings={"recipient": [_RECIPIENT]},
            declared_roles=("recipient",),
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
    ]
    probe = _coverage_probe(((facet_grants,), (facet_loans,)))
    graph = FakeMintGraphReader()

    assert derive_signals(requests, probe, graph=graph) is None
