"""Pins the exporter's record merge, which folds a pinned source record into its target."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from code_index.types import WholeMarker
from scripts.export_graph_entities import export_entities, merge_records


def _record(dcid: str, *, name: str = "Org", aliases: tuple[str, ...] = (), dac_code=None):
    return {
        "dcid": dcid,
        "name": name,
        "kind": "organisation",
        "aliases": list(aliases),
        "dac_code": dac_code,
    }


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [
                _record("Other", name="Other"),
                _record("Source", aliases=("X", "Z")),
                _record("Target", aliases=("Y", "X"), dac_code="896"),
            ],
            [
                _record("Other", name="Other"),
                _record("Target", aliases=("Y", "X", "Z"), dac_code="896"),
            ],
            id="aliases-and-source-removal",
        ),
        pytest.param(
            [_record("Source", dac_code="896"), _record("Target")],
            [_record("Target", dac_code="896")],
            id="source-dac-code-fallback",
        ),
    ],
)
def test_merge_records_preserves_target_fields_and_drops_source(records, expected):
    assert merge_records(records, {"Source": "Target"}) == expected


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([_record("Target")], id="missing-source"),
        pytest.param([_record("Source")], id="missing-target"),
        pytest.param([_record("Source"), _record("Target", name="Renamed")], id="renamed"),
        pytest.param(
            [_record("Source", dac_code="1"), _record("Target", dac_code="2")],
            id="conflicting-dac-code",
        ),
    ],
)
def test_merge_fails_loudly_when_the_pin_no_longer_holds(records):
    with pytest.raises(SystemExit, match="Source -> Target"):
        merge_records(records, {"Source": "Target"})


def _party(
    *,
    axes: tuple[str, ...],
    totals: str | None = None,
    label: str = "total",
    whole: tuple[WholeMarker, ...] = (),
):
    return SimpleNamespace(kind="total", axes=axes, totals=totals, label=label, whole=whole)


def _index(**parties):
    return SimpleNamespace(parties=parties)


class _Reader:
    def __init__(self, *, arcs, dcids_by_type=None, truncated_types=()):
        self.arcs = arcs
        self.dcids_by_type = dcids_by_type or {
            "Organization": ("AfricaFinanceCorporation", "Africa_Finance_Corporation"),
            "DACRecipientUnallocatedByCountry": (),
            "DACUnspecifiedParty": (),
        }
        self.truncated_types = frozenset(truncated_types)

    def node_subjects(self, graph_type, *, property, max_results):
        assert property == "typeOf"
        return SimpleNamespace(
            dcids=self.dcids_by_type[graph_type], truncated=graph_type in self.truncated_types
        )

    def node_arcs_many(self, dcids, *, properties):
        assert properties == ["typeOf", "name", "alternateName", "dacCode"]
        return {dcid: self.arcs.get(dcid) for dcid in dcids}


def _base_arcs():
    return {
        "AfricaFinanceCorporation": {
            "typeOf": ("Organization",),
            "name": ("Africa Finance Corporation",),
            "alternateName": ("AFC",),
            "dacCode": (),
        },
        "Africa_Finance_Corporation": {
            "typeOf": ("Organization",),
            "name": ("Africa Finance Corporation",),
            "alternateName": (),
            "dacCode": (),
        },
    }


def test_export_entities_adds_origin_and_whole_dac_totals_and_lists_only_codes_of_grouped_ones():
    arcs = _base_arcs() | {
        "ExistingOrganisationTotal": {
            "typeOf": ("Organization",),
            "name": ("Existing organisation total",),
            "alternateName": (),
            "dacCode": (),
        },
        "DACOfficialDonors": {
            "typeOf": ("DACGroup",),
            "name": ("official donors (DAC)",),
            "alternateName": ("Official Donors, Total",),
            "dacCode": ("20005",),
        },
        "DACAllRecipients": {
            "typeOf": ("DACGroup",),
            "name": ("all recipients (DAC)",),
            "alternateName": ("All Recipients, Total",),
            "dacCode": ("10200",),
        },
        "DACPairedTotal": {
            "typeOf": ("DACGroup",),
            "name": ("paired total",),
            "alternateName": ("Paired Total, Total",),
            "dacCode": ("20009", "20001"),
        },
        "DACUncodedPairedTotal": {
            "typeOf": ("Grouping",),
            "name": ("uncoded paired total",),
            "alternateName": (),
            "dacCode": (),
        },
    }
    reader = _Reader(
        arcs=arcs,
        dcids_by_type={
            "Organization": (
                "AfricaFinanceCorporation",
                "Africa_Finance_Corporation",
                "ExistingOrganisationTotal",
            ),
            "DACRecipientUnallocatedByCountry": (),
            "DACUnspecifiedParty": (),
        },
    )
    index = _index(
        DACOfficialDonors=_party(axes=("origin",), label="official donors (DAC)"),
        DACAsia=_party(axes=("destination",)),
        DACAllRecipients=_party(
            axes=("destination",),
            label="all recipients (DAC)",
            whole=(WholeMarker("provenance/OECDDAC2A", ("official", "residual"), (1984, 2024)),),
        ),
        DACPairedTotal=_party(axes=("origin",), totals="groups/Paired", label="paired total"),
        DACUncodedPairedTotal=_party(
            axes=("origin",), totals="groups/Uncoded", label="uncoded paired total"
        ),
        ExistingOrganisationTotal=_party(axes=("origin",), label="Existing organisation total"),
    )

    records = export_entities(reader, index=index)

    assert records == [
        _record("Africa_Finance_Corporation", name="Africa Finance Corporation", aliases=("AFC",)),
        {
            "dcid": "DACAllRecipients",
            "name": "all recipients (DAC)",
            "kind": "aggregate",
            "aliases": ["All Recipients, Total"],
            "dac_code": "10200",
        },
        {
            "dcid": "DACOfficialDonors",
            "name": "official donors (DAC)",
            "kind": "aggregate",
            "aliases": ["Official Donors, Total"],
            "dac_code": "20005",
        },
        {
            "dcid": "DACPairedTotal",
            "name": "DACPairedTotal",
            "kind": "aggregate",
            "aliases": ["20009"],
            "dac_code": "20001",
        },
        _record("ExistingOrganisationTotal", name="Existing organisation total"),
    ]


@pytest.mark.parametrize(
    ("reader", "index", "match"),
    [
        pytest.param(
            _Reader(arcs=_base_arcs(), truncated_types=("Organization",)),
            _index(),
            "Organization population truncated",
            id="truncated-population",
        ),
        pytest.param(
            _Reader(arcs=_base_arcs()),
            _index(DACOfficialDonors=_party(axes=("origin",), label="official donors (DAC)")),
            "DACOfficialDonors is absent",
            id="missing-selected-total",
        ),
        pytest.param(
            _Reader(
                arcs=_base_arcs()
                | {
                    "DACOfficialDonors": {
                        "typeOf": ("Country",),
                        "name": ("official donors (DAC)",),
                    }
                }
            ),
            _index(DACOfficialDonors=_party(axes=("origin",), label="official donors (DAC)")),
            "DACOfficialDonors is not a DACGroup",
            id="wrong-type",
        ),
        pytest.param(
            _Reader(
                arcs=_base_arcs() | {"DACOfficialDonors": {"name": ("official donors (DAC)",)}}
            ),
            _index(DACOfficialDonors=_party(axes=("origin",), label="official donors (DAC)")),
            "DACOfficialDonors is not a DACGroup",
            id="missing-type",
        ),
        pytest.param(
            _Reader(arcs=_base_arcs() | {"DACOfficialDonors": {"typeOf": ("DACGroup",)}}),
            _index(DACOfficialDonors=_party(axes=("origin",), label="official donors (DAC)")),
            "name property is empty",
            id="missing-name",
        ),
        pytest.param(
            _Reader(
                arcs=_base_arcs()
                | {
                    "DACOfficialDonors": {
                        "typeOf": ("DACGroup",),
                        "name": ("official donor total (DAC)",),
                    }
                }
            ),
            _index(DACOfficialDonors=_party(axes=("origin",), label="official donors (DAC)")),
            "expected 'official donors \\(DAC\\)'",
            id="renamed-total",
        ),
        pytest.param(
            _Reader(
                arcs=_base_arcs()
                | {"GroupedTotal": {"typeOf": ("Grouping",), "name": ("grouped total renamed",)}}
            ),
            _index(GroupedTotal=_party(axes=("origin",), totals="groups/G", label="grouped total")),
            "expected 'grouped total'",
            id="renamed-grouped-total",
        ),
    ],
)
def test_export_entities_refuses_incomplete_or_invalid_donor_totals(reader, index, match):
    with pytest.raises(SystemExit, match=match):
        export_entities(reader, index=index)
