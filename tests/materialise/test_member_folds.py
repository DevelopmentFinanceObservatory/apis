"""Each fact a source records per member provenance reads, for a declared dataset's id or for
one of its members, as the dataset's members fold it (`materialise.member_folds`).

`dataset/Ledger` holds its series under `provenance/LedgerNorth` and `provenance/LedgerSouth`,
`dataset/Solo` under one member, and `provenance/Plain` is its own sole member
(`tests.materialise._ledger`). Every reader past the read boundary holds the dataset's id, so
each fact is the members' fold, whichever of the two ids a reader holds.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

from dc_sdmx.results import observation_schema
from materialise import member_folds
from materialise.columnar import dataset_cells, held_back_mask
from materialise.continued_series import ContinuedSeries
from materialise.facets import (
    FacetBinding,
    _probe_provenance_refresh,
    declared_default_policies,
    read_as_datasets,
    select_facet,
)
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.held_runs import read_runs
from materialise.member_folds import (
    dataset_name,
    dataset_origin,
    entry_licence,
    is_projection,
    row_licence,
    status_of,
)
from materialise.recipes import RecipeNotDeclaredError, resolve_recipe
from materialise.reported_share import fold_observations, reported_share
from origin_catalogue import bound_catalogue
from origin_catalogue.catalogue import CardFacts, Licence
from origin_catalogue.origins import BASE, DFO
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    dev_provenance_nodes,
    documenting,
    provenance_row,
)

from ._ledger import LEDGER, NORTH, PLAIN, SOLO, SOLO_MEMBER, SOUTH, bound_ledger_index
from .conftest import FakeGraphReader

_STAT_VAR = "dcid:Amount_Ledger"
_ESTIMATED = "Estimated"
_CC_BY = Licence(type="CCBY4", label="CC BY 4.0", url="https://example.org/cc-by")


@pytest.fixture(autouse=True)
def _ledger_index():
    with bound_ledger_index():
        yield


def _row(provenance: str, *, name: str, source: str = "source/Ledger", licence=_CC_BY):
    return dataclasses.replace(
        provenance_row(provenance, DFO, source=source, name=name), licence=licence
    )


def _facet(provenance: str, *, method: str | None = None, latest: str = "2015") -> Facet:
    return Facet(
        facet_id=f"facet-{provenance.rsplit('/', 1)[-1]}-{method}",
        provenance_id=provenance,
        measurement_method=method,
        unit="USDollar",
        obs_count=1,
        earliest_date="2015",
        latest_date=latest,
        observation_period="P1Y",
    )


@pytest.mark.parametrize(
    ("rows", "ref", "name", "source", "licence"),
    [
        pytest.param(
            [_row(NORTH, name="Ledger register"), _row(SOUTH, name="Ledger register")],
            LEDGER,
            "Ledger register",
            "source/Ledger",
            _CC_BY,
            id="members-agree",
        ),
        pytest.param(
            [_row(NORTH, name="Ledger register"), _row(SOUTH, name="Ledger register")],
            SOUTH,
            "Ledger register",
            "source/Ledger",
            _CC_BY,
            id="a-member-reads-as-its-dataset",
        ),
        pytest.param(
            [
                _row(NORTH, name="North books", source="source/North"),
                _row(SOUTH, name="South books", licence=Licence()),
            ],
            LEDGER,
            "Ledger",
            None,
            None,
            id="members-differ-and-the-name-is-the-declared-label",
        ),
        pytest.param(
            [_row(NORTH, name="Ledger register")],
            LEDGER,
            "Ledger register",
            "source/Ledger",
            _CC_BY,
            id="an-unlisted-member-adds-nothing",
        ),
        pytest.param(
            [_row(SOLO_MEMBER, name="Solo books")],
            SOLO,
            "Solo books",
            "source/Ledger",
            _CC_BY,
            id="one-member",
        ),
        pytest.param(
            [_row(PLAIN, name="Plain table")],
            PLAIN,
            "Plain table",
            "source/Ledger",
            _CC_BY,
            id="plain-table",
        ),
    ],
)
def test_catalogue_row_facts_are_the_ones_every_listed_member_shares(
    rows, ref, name, source, licence
):
    """The catalogue lists member provenances only, so a dataset's name, source, Data Commons
    and licence are the ones its listed members' rows share, and its name falls back to its
    declared label."""
    catalogue = FakeCatalogue(
        provenances={row.id: row for row in rows},
        card_facts={row.id: CardFacts(license_type=row.licence.label) for row in rows},
    )

    origin = dataset_origin(ref, catalogue=catalogue)

    assert dataset_name(ref, catalogue=catalogue) == name
    assert (origin[0].ref if origin is not None else None) == source
    assert row_licence(ref, catalogue=catalogue) == licence
    assert entry_licence(ref, catalogue=catalogue) == (licence.label if licence else None)


def test_members_of_two_data_commons_share_no_origin():
    rows = [
        _row(NORTH, name="Ledger"),
        dataclasses.replace(_row(SOUTH, name="Ledger"), data_commons=BASE),
    ]
    catalogue = FakeCatalogue(provenances={row.id: row for row in rows})

    assert dataset_origin(LEDGER, catalogue=catalogue) is None


def test_a_dataset_projects_under_every_method_any_member_declares(
    monkeypatch: pytest.MonkeyPatch,
):
    """North declares `Estimated` a projection method and South declares none. The dataset's
    rows carry its id, and its held runs read each facet's member provenance, so both read the
    members' union: a South row under `Estimated` is held back and its run is projected."""
    declared = {NORTH: frozenset({_ESTIMATED})}
    monkeypatch.setattr(
        member_folds,
        "declared_projection_methods",
        lambda provenance: declared.get(provenance, frozenset()),
    )
    batch = pa.RecordBatch.from_pydict(
        {
            "provenance": [SOUTH, SOUTH, PLAIN],
            "measurementMethod": [_ESTIMATED, None, _ESTIMATED],
            "TIME_PERIOD": ["2026", "2024", "2026"],
        }
    )
    rows = pa.Table.from_batches([dataset_cells(batch)])
    south = _facet(SOUTH, method=_ESTIMATED, latest="2030")

    with bound_catalogue(documenting(_unmarked((NORTH, SOUTH)))):
        mask = held_back_mask(rows, include=frozenset({"partial"})).to_pylist()
        (run,) = read_runs([([south], {south.facet_id or ""})])

    assert [is_projection(ref, _ESTIMATED) for ref in (LEDGER, NORTH, SOUTH, PLAIN)] == [
        True,
        True,
        True,
        False,
    ]
    assert mask == [True, False, False]
    assert (run.kind, run.source) == ("projected", LEDGER)


@pytest.mark.parametrize(
    ("reported", "status"),
    [
        pytest.param({NORTH, SOUTH}, "reported", id="every-member-reports"),
        pytest.param({NORTH}, None, id="one-member-reports"),
    ],
)
def test_a_dataset_reports_its_methodless_values_when_every_member_does(
    monkeypatch: pytest.MonkeyPatch, reported, status
):
    """A method-less value is reported under a provenance the method-status file lists. A
    dataset's rows carry its id, so its values are reported only when every member is
    listed, and a member's facet reads the same status as the dataset's row."""
    monkeypatch.setattr(member_folds, "reported_provenances", lambda: frozenset(reported))

    assert {status_of(None, ref) for ref in (LEDGER, NORTH, SOUTH)} == {status}


def _unmarked(provenances: tuple[str, ...]) -> dict[str, dict[str, tuple[str, ...]]]:
    """Each of `provenances` as the catalogue documents CRS, with no partial mark."""
    crs = dev_provenance_nodes()["provenance/OECDCreditorReportingSystem"]
    node = {prop: values for prop, values in crs.items() if prop != "partialObservationDate"}
    return dict.fromkeys(provenances, node)


@pytest.mark.parametrize(
    ("dates", "ref", "token"),
    [
        pytest.param(
            {LEDGER: "2026-03-01", NORTH: "2026-05-01", SOUTH: "2026-04-01"},
            LEDGER,
            "2026-05-01",
            id="a-member-newer-than-the-node",
        ),
        pytest.param(
            {LEDGER: "2026-06-01", NORTH: "2026-05-01"}, LEDGER, "2026-06-01", id="the-node-newest"
        ),
        pytest.param({NORTH: "2026-05-01"}, LEDGER, "2026-05-01", id="a-node-with-no-date"),
        pytest.param({SOLO_MEMBER: "2026-02-01"}, SOLO, "2026-02-01", id="one-member"),
        pytest.param({PLAIN: "2026-01-01"}, PLAIN, "2026-01-01", id="plain-table"),
    ],
)
def test_a_dataset_is_as_fresh_as_the_newest_of_its_node_and_its_members(dates, ref, token):
    graph = FakeGraphReader(
        arcs={dcid: {"lastDataRefreshDate": (date,)} for dcid, date in dates.items()}
    )

    refresh = _probe_provenance_refresh([ref], graph=graph)

    assert refresh[ref].last_data_refresh_date == token
    assert len(graph.node_arcs_many_calls) == 1


_DAC = "ConversionRecipeDAC"
_IMF = "ConversionRecipeIMF"


@pytest.mark.parametrize(
    ("arcs", "ref", "recipe"),
    [
        pytest.param({LEDGER: (_IMF,), NORTH: (_DAC,), SOUTH: (_DAC,)}, LEDGER, _IMF, id="node"),
        pytest.param({NORTH: (_DAC,), SOUTH: (_DAC,)}, LEDGER, _DAC, id="members-agree"),
        pytest.param({NORTH: (_DAC,), SOUTH: (_IMF,)}, LEDGER, None, id="members-differ"),
        pytest.param({NORTH: (_DAC,)}, LEDGER, None, id="a-member-names-none"),
        pytest.param({SOLO_MEMBER: (_DAC,)}, SOLO, _DAC, id="one-member"),
        pytest.param({}, PLAIN, None, id="plain-table-names-none"),
    ],
)
def test_a_dataset_converts_under_its_nodes_recipe_else_the_one_its_members_share(
    arcs, ref, recipe
):
    graph = FakeGraphReader(
        arcs={
            dcid: {"typeOf": ("Provenance",), "conversionRecipe": ids} for dcid, ids in arcs.items()
        }
    )
    own = graph.node_arcs_many([ref], properties=("conversionRecipe", "isPartOf"))[ref]

    if recipe is None:
        with pytest.raises(RecipeNotDeclaredError) as refused:
            resolve_recipe(graph, provenance_id=ref, provenance_arcs=own)
        assert refused.value.cause == "recipe_undeclared"
    else:
        assert resolve_recipe(graph, provenance_id=ref, provenance_arcs=own).recipe_id == recipe


@pytest.mark.parametrize("declared", [NORTH, LEDGER])
def test_a_default_naming_a_member_provenance_elects_its_dataset(declared):
    """The StatVar declares a provenance default by a member's id or the dataset's. The
    candidates carry the dataset's id, so the default elects the dataset."""
    policy = FacetDefaultPolicy(
        resolved_ref=declared,
        policy_node="defaultProvenance",
        policy_version="v1",
        selection_reason="declared",
    )
    graph = FakeGraphReader(policies={(_STAT_VAR, "provenance"): policy})
    unbound = {"provenance": FacetBinding(state="unbound")}

    selection = select_facet(
        "provenance",
        FacetBinding(state="unbound"),
        candidates=read_as_datasets([_facet(PLAIN), _facet(SOUTH)]),
        graph=graph,
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(unbound, graph=graph, stat_var=_STAT_VAR),
    )

    assert (selection.ref, selection.basis) == (LEDGER, "declared_default")


def _raw_rows(provenances_and_years: list[tuple[str, str, str]]) -> pa.Table:
    """Raw observation rows, one per `(provenance, recipient, year)`, each worth 1 US dollar,
    with each `provenance` cell read as its dataset (`columnar.dataset_cells`)."""
    schema = observation_schema(("recipient",))
    rows = [
        {
            "variableMeasured": _STAT_VAR,
            "recipient": recipient,
            "TIME_PERIOD": year,
            "OBS_VALUE": "1",
            "unit": "USDollar",
            "provenance": provenance,
        }
        for provenance, recipient, year in provenances_and_years
    ]
    batch = pa.RecordBatch.from_pylist(rows, schema=schema)
    return pa.Table.from_batches([dataset_cells(batch)])


def test_a_reported_share_counts_every_member_of_its_dataset_together():
    """North reports two recipients for 2024 and one for 2025, South two and one. The rows
    carry the dataset's id, so one total holds both members: two of four reported."""
    rows = _raw_rows(
        [
            (NORTH, "country/KEN", "2024"),
            (NORTH, "country/UGA", "2024"),
            (NORTH, "country/KEN", "2025"),
            (SOUTH, "country/TZA", "2024"),
            (SOUTH, "country/RWA", "2024"),
            (SOUTH, "country/TZA", "2025"),
        ]
    )

    totals = fold_observations(rows, stat_var=_STAT_VAR, roles=("recipient",))
    share = reported_share(list(totals.values()), year=2025)

    assert list(totals) == [(LEDGER, _STAT_VAR)]
    assert share is not None and (share.share, share.against) == (0.5, 2024)


def test_a_continuation_scoped_to_a_dataset_admits_the_rows_of_every_member():
    """A trace names the dataset's id on its provenance selection, and every member's rows
    carry the id, so the continuation admits the rows of both."""
    rows = _raw_rows([(NORTH, "country/KEN", "2026"), (SOUTH, "country/TZA", "2026")])
    continued = ContinuedSeries(method=_ESTIMATED, continues=None, provenance=LEDGER)

    assert [
        continued.admits(method=_ESTIMATED, provenance=cell)
        for cell in rows["provenance"].to_pylist()
    ] == [True, True]
