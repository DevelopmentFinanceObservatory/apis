"""`resolved_stat_var_dcids` on `kind: spec_id` asserts that each posted StatVar is a member
of the identity's family, before antichain reduction.

The mint and the preview refuse `member_not_resolvable` with one issue per posted non-member,
and a refused mint writes no registry row. A member the root dominates is accepted and not
served. The list narrows nothing. Omitted members are served, and an empty list behaves as an
omitted field. The composed app covers each
resolution path: the graph walk, the code index, the pinned arm, a registry miss with a
sidecar, and a registry hit with and without scope surgery.
"""

from __future__ import annotations

import httpx
import pytest

from code_index.query import CodeIndex
from code_index.types import IndexStamp, StatVarRow
from materialise.graph import Facet
from spec_registry.identity import recompute_identity

from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import described_identity, pinned_identity
from .conftest import FakeDatasetStore, wire_definition_for
from .test_definition_door import _client, _coverage_service

_RECIPIENT = "country/KEN"
_ROOT = "Count_Person"
_MALE = "Count_Person_Male"
_FEMALE = "Count_Person_Female"
_HOUSEHOLD = "Count_Household"
_FIELD = "/input/resolved_stat_var_dcids"

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _arcs(*, population_type: str = "Person", gender: str | None = None) -> dict:
    arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": (population_type,),
        "measuredProperty": ("count",),
        "observationProperties": ("recipient",),
    }
    if gender is not None:
        arcs["constraintProperties"] = ("gender",)
        arcs["gender"] = (gender,)
    return arcs


def _graph(*, with_root: bool = True) -> FakeMintGraphReader:
    """A `Person`/`count` family beside an unrelated `Household` StatVar.

    With the root, the identity resolves to `Count_Person` alone, since the root dominates the
    gendered members. Without it, the identity resolves to the two gendered members.
    """
    arcs = {
        _MALE: _arcs(gender="Male"),
        _FEMALE: _arcs(gender="Female"),
        _HOUSEHOLD: _arcs(population_type="Household"),
    }
    if with_root:
        arcs[_ROOT] = _arcs()
    return FakeMintGraphReader(
        arcs=arcs,
        facets={(stat_var, _RECIPIENT): [_FACET] for stat_var in (_ROOT, _MALE, _FEMALE)},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def _sdmx_transport(stat_vars: tuple[str, ...]) -> httpx.MockTransport:
    def _respond(request: httpx.Request) -> httpx.Response:
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_RECIPIENT},USD,GrantEquivalent,"
            "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
            for stat_var in stat_vars
            if stat_var in str(request.url).replace(f"{stat_var}_", "")
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(_respond)


def _family_index() -> CodeIndex:
    """Carries the `Person`/`count` five-tuple, so resolution reads the index and skips the walk."""

    def _row(dcid: str, constraints: dict[str, str]) -> StatVarRow:
        return StatVarRow(
            dcid=dcid,
            population_type="Person",
            measured_property="count",
            stat_type="measuredValue",
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=("recipient",),
            constraint_properties=constraints,
            provenances=("OECD_DAC2A",),
        )

    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=(),
        statvars=(
            _row(_ROOT, {}),
            _row(_MALE, {"gender": "Male"}),
            _row(_FEMALE, {"gender": "Female"}),
        ),
    )


def _described(*, stat_type: str | None = None) -> dict:
    shape = {"kind": "statvar_schema", "population_type": "Person", "measured_property": "count"}
    if stat_type is not None:
        shape["stat_type"] = stat_type
    return described_identity(
        shape=shape,
        slots=[{"property": "gender", "binding": {"state": "unbound"}}],
        entities={"recipient": [_RECIPIENT]},
    )


def _body(
    document: dict,
    *,
    selection: list[str] | None,
    sidecar: bool = True,
    scope: dict | None = None,
    preview: bool = False,
) -> dict:
    spec_id = recompute_identity(document).spec_id
    input_: dict = {"kind": "spec_id", "spec_id": spec_id}
    if sidecar:
        input_["definition"] = wire_definition_for(spec_id=spec_id, identity=document)
    if selection is not None:
        input_["resolved_stat_var_dcids"] = selection
    if scope is not None:
        input_["scope"] = scope
    body: dict = {"schema_version": 1, "input": input_}
    if preview:
        body["preview"] = {"limit": 10}
    else:
        body["options"] = {"bundle": False}
    return body


def _mint_client(
    *,
    graph: FakeMintGraphReader | None = None,
    store: FakeDatasetStore | None = None,
    code_index: CodeIndex | None = None,
    stat_vars: tuple[str, ...] = (_ROOT, _MALE, _FEMALE),
):
    return _client(
        graph=graph if graph is not None else _graph(),
        store=store,
        sdmx_transport=_sdmx_transport(stat_vars),
        code_index=code_index,
        coverage=_coverage_service(stat_vars=stat_vars, recipient=_RECIPIENT),
    )


def _assert_refused(resp: httpx.Response, *, issues: list[dict]) -> None:
    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "member_not_resolvable"
    assert problem["retryable"] is False
    assert problem["issues"] == issues
    for issue in issues:
        assert issue["member"] in problem["detail"]


def _assert_no_registry_row(store: FakeDatasetStore) -> None:
    assert store.state.spec_registry == {}
    assert store.state.spec_definition == {}


def _resolved_refs(manifest: dict) -> list[str]:
    return [entry["ref"] for entry in manifest["trace"]["resolved_stat_vars"]]


def test_a_selection_of_members_mints_what_an_unselected_mint_mints():
    document = _described()

    unselected = _mint_client().post("/api/datasets", json=_body(document, selection=None))
    selected = _mint_client().post("/api/datasets", json=_body(document, selection=[_ROOT]))

    assert unselected.status_code == 200, unselected.text
    assert selected.status_code == 200, selected.text
    assert _resolved_refs(selected.json()) == [f"dcid:{_ROOT}"]
    assert selected.json()["definition"]["spec_id"] == unselected.json()["definition"]["spec_id"]
    assert selected.json()["content_version"] == unselected.json()["content_version"]


@pytest.mark.parametrize(
    "selection", [[_MALE], [], [_FEMALE, _MALE, _FEMALE]], ids=["subset", "empty", "repeated"]
)
def test_a_selection_omitting_members_mints_the_whole_family(selection: list[str]):
    """The list checks membership and narrows nothing. Omitted members are served."""
    document = _described()

    unselected = _mint_client(graph=_graph(with_root=False)).post(
        "/api/datasets", json=_body(document, selection=None)
    )
    selected = _mint_client(graph=_graph(with_root=False)).post(
        "/api/datasets", json=_body(document, selection=selection)
    )

    assert unselected.status_code == 200, unselected.text
    assert selected.status_code == 200, selected.text
    assert sorted(_resolved_refs(selected.json())) == [f"dcid:{_FEMALE}", f"dcid:{_MALE}"]
    assert selected.json()["content_version"] == unselected.json()["content_version"]


def test_a_posted_dcid_outside_the_identity_refuses_and_writes_no_registry_row():
    store = FakeDatasetStore()
    client = _mint_client(store=store)

    resp = client.post("/api/datasets", json=_body(_described(), selection=[_HOUSEHOLD, _ROOT]))

    _assert_refused(resp, issues=[{"pointer": f"{_FIELD}/0", "member": _HOUSEHOLD}])
    _assert_no_registry_row(store)


@pytest.mark.parametrize("preview", [False, True], ids=["mint", "preview"])
def test_a_dcid_prefixed_entry_is_request_invalid_naming_the_prefix(preview: bool):
    """The field carries bare dcids. A prefixed member is refused as malformed before
    resolution, so the problem names the prefix and leaves the member's membership alone."""
    store = FakeDatasetStore()
    client = _mint_client(store=store)

    resp = client.post(
        "/api/datasets",
        json=_body(_described(), selection=[f"dcid:{_ROOT}"], preview=preview),
    )

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("request_invalid", False)
    assert "without the dcid: prefix" in resp.text
    _assert_no_registry_row(store)


@pytest.mark.parametrize(
    "selection", [[_ROOT, _FEMALE], [_FEMALE]], ids=["beside_the_root", "alone"]
)
def test_a_member_the_root_dominates_is_accepted_and_the_root_alone_is_served(
    selection: list[str],
):
    """`Count_Person_Female` belongs to the walked family and the root dominates it, so the
    check accepts it and the mint serves the reduced family, the root alone."""
    document = _described()

    unselected = _mint_client().post("/api/datasets", json=_body(document, selection=None))
    selected = _mint_client().post("/api/datasets", json=_body(document, selection=selection))

    assert unselected.status_code == 200, unselected.text
    assert selected.status_code == 200, selected.text
    assert _resolved_refs(selected.json()) == [f"dcid:{_ROOT}"]
    assert selected.json()["content_version"] == unselected.json()["content_version"]


def test_every_posted_non_member_gets_its_own_issue_and_omitted_members_get_none():
    store = FakeDatasetStore()
    client = _mint_client(graph=_graph(with_root=False), store=store)

    resp = client.post(
        "/api/datasets", json=_body(_described(), selection=[_HOUSEHOLD, _MALE, _ROOT])
    )

    _assert_refused(
        resp,
        issues=[
            {"pointer": f"{_FIELD}/0", "member": _HOUSEHOLD},
            {"pointer": f"{_FIELD}/2", "member": _ROOT},
        ],
    )
    assert _FEMALE not in resp.json()["detail"]
    _assert_no_registry_row(store)


def test_the_index_path_checks_the_selection():
    """The code index carries the five-tuple, so no reverse walk runs and the posted list
    reaches no enumerator. The check refuses a non-member and accepts a member the root
    dominates, serving the root alone."""
    document = _described(stat_type="measuredValue")
    graph = _graph()
    store = FakeDatasetStore()
    client = _mint_client(graph=graph, store=store, code_index=_family_index())

    refused = client.post("/api/datasets", json=_body(document, selection=[_ROOT, _HOUSEHOLD]))

    _assert_refused(refused, issues=[{"pointer": f"{_FIELD}/1", "member": _HOUSEHOLD}])
    _assert_no_registry_row(store)
    assert graph.calls["node_subjects"] == 0

    for selection in ([_ROOT], [_ROOT, _MALE], [_MALE]):
        minted = client.post("/api/datasets", json=_body(document, selection=selection))

        assert minted.status_code == 200, (selection, minted.text)
        assert _resolved_refs(minted.json()) == [f"dcid:{_ROOT}"]
    assert graph.calls["node_subjects"] == 0


@pytest.mark.parametrize("registered", [False, True], ids=["sidecar", "registry_hit"])
def test_preview_checks_the_selection(registered: bool):
    document = _described()
    store = FakeDatasetStore()
    client = _mint_client(store=store)
    if registered:
        minted = client.post("/api/datasets", json=_body(document, selection=None))
        assert minted.status_code == 200, minted.text
    rows_before = dict(store.state.spec_registry)

    refused = client.post(
        "/api/datasets", json=_body(document, selection=[_HOUSEHOLD], preview=True)
    )

    _assert_refused(refused, issues=[{"pointer": f"{_FIELD}/0", "member": _HOUSEHOLD}])
    assert store.state.spec_registry == rows_before

    accepted = client.post("/api/datasets", json=_body(document, selection=[_ROOT], preview=True))

    assert accepted.status_code == 200, accepted.text
    assert [member["stat_var"] for member in accepted.json()["members"]] == [f"dcid:{_ROOT}"]


def test_a_pinned_arm_accepts_only_its_own_stat_var():
    document = pinned_identity(stat_var=_ROOT, entities={"recipient": [_RECIPIENT]})
    store = FakeDatasetStore()
    client = _mint_client(store=store)

    refused = client.post("/api/datasets", json=_body(document, selection=[_MALE]))

    _assert_refused(refused, issues=[{"pointer": f"{_FIELD}/0", "member": _MALE}])
    _assert_no_registry_row(store)

    minted = client.post("/api/datasets", json=_body(document, selection=[_ROOT]))

    assert minted.status_code == 200, minted.text
    assert _resolved_refs(minted.json()) == [f"dcid:{_ROOT}"]


def test_a_registry_hit_checks_the_selection():
    document = _described()
    store = FakeDatasetStore()
    client = _mint_client(store=store)
    minted = client.post("/api/datasets", json=_body(document, selection=None))
    assert minted.status_code == 200, minted.text
    rows_before = dict(store.state.spec_registry)

    refused = client.post(
        "/api/datasets", json=_body(document, selection=[_ROOT, _HOUSEHOLD], sidecar=False)
    )
    _assert_refused(refused, issues=[{"pointer": f"{_FIELD}/1", "member": _HOUSEHOLD}])
    assert store.state.spec_registry == rows_before

    accepted = client.post("/api/datasets", json=_body(document, selection=[_ROOT], sidecar=False))
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["content_version"] == minted.json()["content_version"]


@pytest.mark.parametrize(
    "scope",
    [
        {"window": {"kind": "range", "start_year": 2020, "end_year": 2020}},
        {"entities": {"recipient": ["country/UGA"]}},
    ],
    ids=["window", "entities"],
)
def test_a_refused_selection_on_a_registry_hit_writes_no_surgered_row(scope: dict):
    """Scope surgery mints a second identity. The check runs before that registration."""
    document = _described()
    store = FakeDatasetStore()
    client = _mint_client(store=store)
    minted = client.post("/api/datasets", json=_body(document, selection=None))
    assert minted.status_code == 200, minted.text
    rows_before = set(store.state.spec_registry)
    assert len(rows_before) == 1

    refused = client.post(
        "/api/datasets",
        json=_body(document, selection=[_HOUSEHOLD, _ROOT], sidecar=False, scope=scope),
    )

    _assert_refused(refused, issues=[{"pointer": f"{_FIELD}/0", "member": _HOUSEHOLD}])
    assert set(store.state.spec_registry) == rows_before

    # As a control, the same scope with a selection of members registers the surgered identity.
    client.post(
        "/api/datasets", json=_body(document, selection=[_ROOT], sidecar=False, scope=scope)
    )
    assert len(store.state.spec_registry) == 2


_GDP_PER_CAPITA = "Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita"
_GDP_PER_PERSON = (
    "Amount_EconomicActivity_GrossDomesticProduction_Nominal_AsAFractionOf_Count_Person"
)


def _per_capita_graph() -> FakeMintGraphReader:
    """Nominal GDP StatVars as the graph holds them. The per-capita member names `PerCapita`,
    a node typed only `ProvisionalNode`, beside a sibling over `Count_Person` and forty
    sector StatVars with no denominator. Each sector StatVar is an absence suspect for an
    unresolved-denominator identity, and eight of them exceed the 32-dcid re-probe budget."""

    def _gdp(**extra: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("EconomicActivity",),
            "measuredProperty": ("amount",),
            "statType": ("measuredValue",),
            "measurementQualifier": ("Nominal",),
            "observationProperties": ("recipient",),
            "constraintProperties": ("activitySource",),
            "activitySource": ("GrossDomesticProduction",),
            **extra,
        }

    arcs = {
        _GDP_PER_CAPITA: _gdp(measurementDenominator=("PerCapita",)),
        _GDP_PER_PERSON: _gdp(measurementDenominator=(_ROOT,)),
        "PerCapita": {"typeOf": ("ProvisionalNode",)},
        _ROOT: {"typeOf": ("StatisticalVariable",)},
    }
    for i in range(40):
        arcs[f"Nominal_Amount_GDP_Sector{i:02d}"] = _gdp(
            constraintProperties=("activitySource", "sector"), sector=(f"Sector{i:02d}",)
        )
    return FakeMintGraphReader(
        arcs=arcs,
        facets={(_GDP_PER_CAPITA, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def _per_capita_identity() -> dict:
    return described_identity(
        shape={
            "kind": "statvar_schema",
            "population_type": "EconomicActivity",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
            "measurement_denominator": {"unresolved": True},
        },
        slots=[
            {
                "property": "activitySource",
                "binding": {"state": "value", "ref": "GrossDomesticProduction"},
            }
        ],
        entities={"recipient": [_RECIPIENT]},
    )


def _per_capita_client(store: FakeDatasetStore | None = None):
    return _mint_client(graph=_per_capita_graph(), store=store, stat_vars=(_GDP_PER_CAPITA,))


def test_an_unresolved_denominator_identity_mints_its_per_capita_member_listed_or_not():
    """`PerCapita` reads as unresolved, as Search reads it, and the per-capita root leaves
    the unread sector StatVars unable to change the members."""
    document = _per_capita_identity()

    unselected = _per_capita_client().post("/api/datasets", json=_body(document, selection=None))
    selected = _per_capita_client().post(
        "/api/datasets", json=_body(document, selection=[_GDP_PER_CAPITA])
    )

    assert unselected.status_code == 200, unselected.text
    assert selected.status_code == 200, selected.text
    assert _resolved_refs(selected.json()) == [f"dcid:{_GDP_PER_CAPITA}"]
    assert selected.json()["content_version"] == unselected.json()["content_version"]


def test_a_resolved_denominator_sibling_is_outside_an_unresolved_denominator_identity():
    store = FakeDatasetStore()

    resp = _per_capita_client(store).post(
        "/api/datasets",
        json=_body(_per_capita_identity(), selection=[_GDP_PER_CAPITA, _GDP_PER_PERSON]),
    )

    _assert_refused(resp, issues=[{"pointer": f"{_FIELD}/1", "member": _GDP_PER_PERSON}])
    _assert_no_registry_row(store)
