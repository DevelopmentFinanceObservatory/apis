"""End-to-end tests for a side read as its whole population.

The packaged code index marks "Official Donors, Total" (`DACOfficialDonors`) as the total of
every `official` donor in DAC2a and DAC2b, "Private Donors, Total" (`DACPrivateDonors`) as the
total of every `private` donor in DAC2a, and "All Recipients, Total" (`DACAllRecipients`) as
the total of every `official` and `residual` recipient in DAC2a from 1984 to 2024. "All donors
including private" joins the first two.

Each question goes through the parsed door as an outside client sent it to the `search`
tool, its part copied from the recorded client transcripts of the "all donors" questions
(q06, q12, q17) and of the questions that list donors (q13, q35, q37), with the recipient
moved to Ethiopia where the client asked about Kenya. The graph's search finds
DAC2a's official development assistance member first and the CRS's second, unless a test
says otherwise, and the scripted model elects the first shape it is offered. Coverage holds
series to Ethiopia from France, from the official donors' total and from the private
donors' total in DAC2a, and from France in the CRS.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.departures import OPERATIONS_NOT_RUN
from search.engine.extract import ExtractedEntity, ExtractedOperation
from search.engine.region import (
    candidate_collapse,
    phase_election,
    phase_place_materialise,
    place_roles,
)
from search.engine.region.place_roles import query_role_axes
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.region.state import ResolveState
from search.engine.region.whole_totals import covers_unpublished, whole_reading
from search.engine.retrieval.bound_values import declared_datasets
from search.engine.role_text import DoorRole
from search.models import (
    CandidatesPart,
    DatasetResubmitInput,
    DefinitePart,
    EntityGroup,
    NoDataPart,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    Resolution,
    ResolveRequest,
    ResolveResponse,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    Spec,
    role_ref,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.code_index.aid_identity import CRS_PROVENANCE, IMPUTED_PROVENANCE
from tests.engine._binders import ReplyBinder
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import member_walk_reader, offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import (
    _COMMITMENTS_MEMBER,
    _CRS_MEMBER,
    _DAC2A,
    _DAC2A_MEMBER,
    _IMPUTED_MEMBER,
    _arcs,
    _ScriptedLLM,
    source_catalogue,
)
from tests.fixtures import FakeGraph

_DAC1 = "provenance/OECDDAC1"
_DAC2B = "provenance/OECDDAC2B"
_ETH = "country/ETH"
_FRA = "country/FRA"
_OFFICIAL = "DACOfficialDonors"
_PRIVATE = "DACPrivateDonors"
_ALL_RECIPIENTS = "DACAllRecipients"
_VARIABLE = "official development assistance"


def _graph(
    *,
    crs_first: bool = False,
    variable: str = _VARIABLE,
    members: tuple[str, ...] | None = None,
) -> FakeGraph:
    """A graph whose search finds `members` for `variable`, by default DAC2a's member and
    the CRS's, DAC2a first unless `crs_first`. It holds Ethiopia, France and the three
    whole totals, which the resolver reads "all donors", "all recipients" and "private
    donors" as."""
    if members is None:
        members = (_CRS_MEMBER, _DAC2A_MEMBER) if crs_first else (_DAC2A_MEMBER, _CRS_MEMBER)
    return FakeGraph(
        nodes={
            _ETH: {"label": "Ethiopia", "type": "Country"},
            _FRA: {"label": "France", "type": "Country"},
            **{
                dcid: {"label": dcid, "type": "Organization"}
                for dcid in (_OFFICIAL, _PRIVATE, _ALL_RECIPIENTS)
            },
            _DAC2A: {"label": "OECD DAC2a"},
            CRS_PROVENANCE: {"label": "OECD Creditor Reporting System (CRS)"},
            IMPUTED_PROVENANCE: {"label": "ONE Imputed Multilateral Sectors"},
            **{
                dcid: {"label": dcid, "type": "StatisticalVariable", "arcs": _arcs(dcid)}
                for dcid in members
            },
        },
        detect={
            variable: {
                "svs": list(members),
                "cosine_scores": [0.9 - 0.05 * rank for rank in range(len(members))],
            }
        },
        resolve={"Ethiopia": _ETH, "France": _FRA},
        obs={},
    )


def _coverage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    official: bool = True,
    years: tuple[int, ...] = (2019, 2020, 2021),
) -> None:
    """DAC2a holds series to Ethiopia from France, from the private donors' total and,
    unless told otherwise, from the official donors' total, and from France to every
    recipient; the CRS holds France's series to Ethiopia."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    span = list(years)
    patch_coverage(
        monkeypatch,
        [
            *([(_DAC2A_MEMBER, "dac2a", _OFFICIAL, _ETH, span)] if official else []),
            (_DAC2A_MEMBER, "dac2a", _PRIVATE, _ETH, span),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, span),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ALL_RECIPIENTS, span),
            (_CRS_MEMBER, "crs", _FRA, _ETH, span),
        ],
        role_names=dict.fromkeys((_DAC2A_MEMBER, _CRS_MEMBER), names),
        facet_metadata={
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
        },
    )


def _parsed(part: dict[str, Any]) -> ResolveRequest:
    """The parsed request data-mcp's `search` tool posts for one part a client sent it, in
    the tool's own field names (`data_mcp.search_tool._wire_part`)."""
    wire: dict[str, Any] = {"variable_text": part["variable"]}
    if "entities" in part:
        wire["entities"] = [
            {
                "text": entity["text"],
                **({"role_text": entity["role"]} if "role" in entity else {}),
                **({"members": entity["members"]} if "members" in entity else {}),
            }
            for entity in part["entities"]
        ]
    if "time" in part:
        wire["time_text"] = part["time"]
    for key in ("open_roles", "operations", "sources"):
        if key in part:
            wire[key] = part[key]
    return ResolveRequest(input=ParsedInput(parts=[ParsedPart.model_validate(wire)]))


def _resolve(
    part: dict[str, Any], *, drafted: tuple[SlotBindingDraft, ...] = (), **graph: Any
) -> ResolveResponse:
    return offline_resolve(
        _parsed(part),
        graph=_graph(variable=part["variable"], **graph),
        llm=_ScriptedLLM(part["variable"], drafted=drafted),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


def _spec(response: ResolveResponse) -> Spec:
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part.spec


def _part(response: ResolveResponse) -> DefinitePart:
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part


def _whole_read(part: DefinitePart) -> tuple[str, str | None] | None:
    """The `whole_total_read` departure's binding and the request words it asks, if any."""
    found = [
        (departure.binding_ref, departure.asked)
        for departure in part.departures
        if departure.code == "whole_total_read"
    ]
    assert len(found) <= 1, found
    return found[0] if found else None


def _available(part: DefinitePart) -> list[dict[str, Any] | None]:
    """The params of each `whole_total_available` warning the part carries."""
    return [warning.params for warning in part.warnings if warning.code == "whole_total_available"]


def _origin(spec: Spec) -> list[str] | None:
    return spec.definition.identity.entities.get("origin")


def _group(spec: Spec, role: str) -> dict[str, Any] | None:
    groups = spec.resolved.entity_groups or {}
    group = groups.get(role)
    return None if group is None else group.model_dump(mode="json", exclude_none=True)


_TO_ETHIOPIA = {"role": "to", "text": "Ethiopia"}
_FROM_FRANCE = {"role": "from", "text": "France"}
# DAC2a publishes official, private and residual donors, so "Private donors" is outside the
# official donors' total there.
_OFFICIAL_GROUP = {
    "label": "official donors (DAC)",
    "side_classes": ["official", "private", "residual"],
    "total": _OFFICIAL,
    "whole_of": ["official"],
}
_CRS_NAMED = [{"ref": CRS_PROVENANCE}]


@pytest.mark.parametrize(
    ("part", "asked"),
    [
        pytest.param(
            {"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE},
            None,
            id="no-donor-named",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "official donors"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "official donors",
            id="official-donors-named",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "All Donors, Total"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "All Donors, Total",
            id="all-donors-total-named",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "Official Donors, Total"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "Official Donors, Total",
            id="the-totals-label-named",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "operations": [{"phrase": "total aid from all donors", "verb": "aggregate"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            None,
            id="aggregate-over-no-axis",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "operations": [{"phrase": "total from all donors", "verb": "report"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            None,
            id="report-over-no-axis",
        ),
    ],
)
def test_a_totalled_or_whole_named_donor_side_binds_the_official_donors_total(
    monkeypatch: pytest.MonkeyPatch, part: dict[str, Any], asked: str | None
):
    """Each "all donors" question an outside client sent reads DAC2a's official donors'
    total on the donor side: no donor named, "official donors", "All Donors, Total" or the
    total's label named, and a sum or report over no axis. The role carries the population
    as its group, the part departs as `whole_total_read` asked in the request's words for
    the population, and no side is listed member by member, so no `whole_total_available`.
    A sum over no axis departs as not applied, as any operation Search does not run."""
    _coverage(monkeypatch)

    answer = _part(_resolve(part))

    spec = answer.spec
    assert _origin(spec) == [_OFFICIAL]
    assert spec.definition.identity.entities["destination"] == [_ETH]
    assert spec.definition.identity.open_roles is None
    assert _group(spec, "origin") == _OFFICIAL_GROUP
    assert _whole_read(answer) == ("origin:", asked)
    assert [
        departure.asked
        for departure in answer.departures
        if departure.code == "operation_not_applied"
    ] == [op["phrase"] for op in part.get("operations", ()) if op["verb"] in OPERATIONS_NOT_RUN]
    assert _available(answer) == []


@pytest.mark.parametrize(
    "part",
    [
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="open-donors",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["from"],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="open-from",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "operations": [{"over": "origin", "phrase": "top five donors", "verb": "rank"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="donors-ranked",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "operations": [{"over": "origin", "phrase": "top five donors", "verb": "rank"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="donors-ranked-without-open-roles",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "operations": [{"over": "origin", "phrase": "how many donors", "verb": "report"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="donors-counted-in-a-report",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "operations": [
                    {"over": "origin", "phrase": "share of each donor", "verb": "share_of"}
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="donors-shared",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "operations": [
                    {"over": "origin", "phrase": "from all donors", "verb": "aggregate"}
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="open-donors-summed",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "operations": [
                    {
                        "over": "origin",
                        "phrase": "sum aid from donors giving more than 100 million",
                        "verb": "aggregate",
                    }
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="open-donors-summed-over-a-threshold",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "operations": [
                    {
                        "over": "origin",
                        "phrase": "number of donors giving more than 100 million",
                        "verb": "aggregate",
                    }
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="open-donors-counted",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "operations": [
                    {"over": "origin", "phrase": "total from all donors", "verb": "aggregate"}
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="donors-summed-without-open-roles",
        ),
    ],
)
def test_a_listed_donor_side_keeps_its_members(
    monkeypatch: pytest.MonkeyPatch, part: dict[str, Any]
):
    """A donor side the part opens with no operation, or runs any operation over, lists its
    donors, as the questions asking for the top donors, for how many donors gave and for a
    sum over the donors past a threshold send it: an operation over the side may filter or
    count its members, so no total stands in for them. The identity leaves the donor side
    open and carries no group, and each operation over it that Search does not run departs
    as not applied. DAC2a
    publishes the official donors' total, so the part states it beside the listed donors,
    whether `open_roles` opened the side or an operation over it did."""
    _coverage(monkeypatch)

    answer = _part(_resolve(part))

    spec = answer.spec
    assert _origin(spec) is None
    assert spec.definition.identity.open_roles == ["origin"]
    assert _group(spec, "origin") is None
    assert _whole_read(answer) is None
    assert _available(answer) == [
        {"total": _OFFICIAL, "total_label": "official donors (DAC)", "whole_of": "official"}
    ]
    over_donors = [
        op["phrase"]
        for op in part.get("operations", ())
        if op["over"] == "origin" and op["verb"] in OPERATIONS_NOT_RUN
    ]
    assert [
        departure.asked
        for departure in answer.departures
        if departure.code == "operation_not_applied"
    ] == over_donors


def test_official_donors_read_the_official_total_where_the_crs_holds_private_donors(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "Official donors" names the official donors' total, whose markers hold official
    donors alone. The CRS publishes no such total and holds private donors beside official
    ones, so its sum over every donor cannot stand in for it: the CRS shape binds the total
    as a named party, holds no series from it, and DAC2a answers with the total read whole,
    though the search ranks the CRS first."""
    _coverage(monkeypatch)
    named = {
        "entities": [_TO_ETHIOPIA, {"role": "from", "text": "official donors"}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    answer = _part(_resolve(named, crs_first=True))

    spec = answer.spec
    assert [source.ref for source in spec.resolved.trace.resolved_sources] == [_DAC2A]
    assert _origin(spec) == [_OFFICIAL]
    assert _group(spec, "origin") == _OFFICIAL_GROUP
    assert _whole_read(answer) == ("origin:", "official donors")


@pytest.mark.parametrize("text", ["all donors including private", "all donors", "every donor"])
def test_all_donors_binds_both_totals(monkeypatch: pytest.MonkeyPatch, text: str):
    """Each phrase of the whole total set of official and private donors binds both totals
    on DAC2a, where both are marked, and the role carries the set, labelled by its first
    phrase, with the classes it covers."""
    _coverage(monkeypatch)
    part = {
        "entities": [_TO_ETHIOPIA, {"role": "from", "text": text}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    response = _resolve(part)

    spec = _spec(response)
    assert _origin(spec) == [_OFFICIAL, _PRIVATE]
    assert _group(spec, "origin") == {
        "label": "all donors including private",
        "side_classes": ["official", "private", "residual"],
        "whole_of": ["official", "private"],
    }
    assert "group_read_as_members" not in [
        departure.code for departure in response.parts[0].departures
    ]
    assert "from all donors including private for" in (spec.resolved.rendered_sentence or "")


_CRS_DONORS_COVERED = (
    "OECD Creditor Reporting System (CRS) publishes no total of every donor, so this covers "
    "every donor it holds with data, private donors among them."
)


@pytest.mark.parametrize(
    ("part", "side", "text"),
    [
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "all donors"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "origin",
            f"You asked for all donors. {_CRS_DONORS_COVERED}",
            id="all-donors-crs-first",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "all donors"}],
                "sources": _CRS_NAMED,
                "time": "2021",
                "variable": _VARIABLE,
            },
            "origin",
            f"You asked for all donors. {_CRS_DONORS_COVERED}",
            id="all-donors-crs-named",
        ),
        pytest.param(
            {
                "entities": [
                    _TO_ETHIOPIA,
                    {"role": "from", "text": "all donors including private"},
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "origin",
            f"You asked for all donors including private. {_CRS_DONORS_COVERED}",
            id="all-donors-including-private-crs-first",
        ),
        pytest.param(
            {
                "entities": [_FROM_FRANCE, {"role": "to", "text": "all recipients"}],
                "sources": _CRS_NAMED,
                "time": "2021",
                "variable": _VARIABLE,
            },
            "destination",
            "You asked for all recipients. OECD Creditor Reporting System (CRS) publishes no "
            "total of every recipient, so this covers every recipient it holds with data, which "
            "includes aid not allocated to a country.",
            id="all-recipients-crs-named",
        ),
    ],
)
def test_a_population_covering_its_side_on_the_crs_reads_as_every_party_it_holds(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    part: dict[str, Any],
    side: str,
    text: str,
):
    """The CRS marks no whole total. Every donor it holds is an official or a private one,
    the classes "all donors" covers, and every recipient a country or an unallocated residual,
    the classes "all recipients" covers, so the side reads as unnamed and the answer covers
    every party the CRS holds there. It departs `whole_total_unpublished` in the request's
    words, and no `role_scope_defaulted` says that nothing was named there."""
    if "sources" in part:
        request.getfixturevalue(source_catalogue.__name__)
    _coverage(monkeypatch)

    answer = _part(_resolve(part, crs_first=True))

    spec = answer.spec
    assert [source.ref for source in spec.resolved.trace.resolved_sources] == [CRS_PROVENANCE]
    assert spec.definition.identity.entities.get(side) is None
    assert _group(spec, side) is None
    assert [
        (departure.binding_ref, departure.text)
        for departure in answer.departures
        if departure.code == "whole_total_unpublished"
    ] == [(role_ref(side), text)]
    noun = "provider" if side == "origin" else "recipient"
    assert not [
        warning
        for warning in answer.warnings
        if warning.code == "role_scope_defaulted"
        and warning.message.startswith(f"no {noun} was named")
    ]


def _crs_and_imputed_answer(monkeypatch: pytest.MonkeyPatch, text: str) -> Any:
    """The one part answering aid from `text` to Ethiopia in 2021, where the bind elects the
    CRS commitments shape and nominates the imputed multilateral shape beside it, and
    coverage holds France's series to Ethiopia on both."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_COMMITMENTS_MEMBER, "crs", _FRA, _ETH, [2021]),
            (_IMPUTED_MEMBER, "imputed", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((_COMMITMENTS_MEMBER, _IMPUTED_MEMBER), names),
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD"),
        },
    )
    part = {
        "entities": [_TO_ETHIOPIA, {"role": "from", "text": text}],
        "time": "2021",
        "variable": "aid",
    }
    (answer,) = offline_resolve(
        _parsed(part),
        graph=_graph(members=(_COMMITMENTS_MEMBER, _IMPUTED_MEMBER), variable="aid"),
        llm=_ScriptedLLM("aid", also_plausible=[1]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        binder=ReplyBinder(),
    ).parts
    return answer


def _no_donor_named_warnings(answer: Any) -> list[Any]:
    return [
        warning
        for warning in answer.warnings
        if warning.code == "role_scope_defaulted"
        and warning.message.startswith("no provider was named")
    ]


def test_a_population_covering_its_side_states_so_beside_each_reading_of_a_candidate_set(
    monkeypatch: pytest.MonkeyPatch,
):
    """Neither the CRS nor the imputed multilateral dataset publishes a donor total, and
    "all donors" covers both datasets' donors, so each reading of the candidate set leaves
    its donor side unnamed. The set departs `whole_total_unpublished` once for each dataset
    its readings come from, and warns no `role_scope_defaulted` that no donor was named."""
    answer = _crs_and_imputed_answer(monkeypatch, "all donors")

    assert isinstance(answer, CandidatesPart), answer
    assert [
        departure.text
        for departure in answer.departures
        if departure.code == "whole_total_unpublished"
    ] == [
        f"You asked for all donors. {_CRS_DONORS_COVERED}",
        "You asked for all donors. ONE Imputed Multilateral Sectors publishes no total "
        "of every donor, so this covers every donor it holds with data.",
    ]
    assert _no_donor_named_warnings(answer) == []


def test_official_donors_covering_the_alternate_alone_refuses_with_the_alternate_as_its_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "Official donors" covers the imputed multilateral dataset's donors, all official,
    and not the CRS's, which include private ones. The elected CRS reading binds
    `DACOfficialDonors` as a named party, which holds no series, and the imputed reading
    leaves its donor side unnamed. As for any party the elected reading holds no data at and
    an alternate does, the part refuses `no_observations` and offers the alternate as its
    match (`only_alternate_has_data`). The match reads its donor side as unnamed, so the
    part departs `whole_total_unpublished` for it as a candidate set does for that reading,
    and warns no `role_scope_defaulted`."""
    origins: dict[tuple[str, ...], list[str] | None] = {}
    named = place_roles._named_wholes

    def recording(state: Any, shape: Any, assignment: Any) -> Any:
        result = named(state, shape, assignment)
        datasets = tuple(row.provenance for row in declared_datasets(shape, get_index()))
        sets = result.sets.get("origin")
        origins[datasets] = None if sets is None else sorted(s.node.dcid for s in sets)
        return result

    monkeypatch.setattr(place_roles, "_named_wholes", recording)

    answer = _crs_and_imputed_answer(monkeypatch, "official donors")

    assert origins == {(CRS_PROVENANCE,): [_OFFICIAL], (IMPUTED_PROVENANCE,): None}
    assert isinstance(answer, NoDataPart), answer
    assert answer.no_data.reason == "no_observations"
    assert answer.no_data.gate == "only_alternate_has_data"
    match = answer.no_data.alternate_match
    assert match is not None
    assert match.definition.identity.entities == {"destination": [_ETH]}
    assert [source.ref for source in match.resolved.trace.resolved_sources] == [IMPUTED_PROVENANCE]
    assert [
        (departure.binding_ref, departure.asked, departure.text)
        for departure in answer.departures
        if departure.code == "whole_total_unpublished"
    ] == [
        (
            "origin:",
            "official donors",
            "You asked for official donors. ONE Imputed Multilateral Sectors publishes no total "
            "of every donor, so the closest match covers every donor it holds with data.",
        )
    ]
    assert _no_donor_named_warnings(answer) == []


def test_an_alternate_match_states_the_total_its_reading_lists_beside(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bind elects the CRS commitments shape for aid to Ethiopia from every donor and
    nominates DAC2a's official development assistance shape beside it. Coverage holds
    series in DAC2a alone, so the part refuses `no_observations` and offers DAC2a's reading
    as its match (`only_alternate_has_data`). That reading lists its donors, and DAC2a
    publishes the official donors' total with a series to Ethiopia, so the part states the
    total, stamped with the match, as the match's definite answer would."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_DAC2A_MEMBER, "dac2a", _OFFICIAL, _ETH, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((_COMMITMENTS_MEMBER, _DAC2A_MEMBER), names),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )
    part = {"entities": [_TO_ETHIOPIA], "open_roles": ["donor"], "time": "2021", "variable": "aid"}

    response = offline_resolve(
        _parsed(part),
        graph=_graph(members=(_COMMITMENTS_MEMBER, _DAC2A_MEMBER), variable="aid"),
        llm=_ScriptedLLM("aid", also_plausible=[1]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        binder=ReplyBinder(),
    )

    assert isinstance(response, Resolution), response
    (answer,) = response.parts
    assert isinstance(answer, NoDataPart), answer
    assert answer.no_data.gate == "only_alternate_has_data"
    match = answer.no_data.alternate_match
    assert match is not None
    assert match.definition.identity.entities == {"destination": [_ETH]}
    assert [
        (warning.spec_id, warning.params)
        for warning in answer.warnings
        if warning.code == "whole_total_available"
    ] == [
        (
            match.spec_id,
            {"total": _OFFICIAL, "total_label": "official donors (DAC)", "whole_of": "official"},
        )
    ]


def test_a_donor_named_alone_binds_no_whole_total(monkeypatch: pytest.MonkeyPatch):
    """ "Private donors" names the private donors' total, which holds private donors alone,
    as a party: it binds as named and carries no whole population."""
    _coverage(monkeypatch)
    part = {
        "entities": [_TO_ETHIOPIA, {"role": "from", "text": "private donors"}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    spec = _spec(_resolve(part))

    assert _origin(spec) == [_PRIVATE]
    assert _group(spec, "origin") is None


@pytest.mark.parametrize(
    "part",
    [
        pytest.param(
            {"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE},
            id="no-donor-named",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA],
                "open_roles": ["donor"],
                "time": "2021",
                "variable": _VARIABLE,
            },
            id="open-donors",
        ),
    ],
)
def test_a_total_holding_no_series_leaves_the_side_to_its_parties(
    monkeypatch: pytest.MonkeyPatch, part: dict[str, Any]
):
    """Where the coverage snapshot holds no DAC2a series from the official donors' total
    to Ethiopia, DAC2a answers from its donors, as with no total marked, and states no
    total beside them, since none would answer."""
    _coverage(monkeypatch, official=False)

    answer = _part(_resolve(part))

    spec = answer.spec
    assert [source.ref for source in spec.resolved.trace.resolved_sources] == [_DAC2A]
    assert _origin(spec) is None
    assert _group(spec, "origin") is None
    assert _available(answer) == []


def test_a_named_whole_population_holding_no_series_refuses_as_a_named_party(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "Official donors" names the official donors' total as a party. Where the coverage
    snapshot holds no series from it to Ethiopia, the part refuses as for any named party
    with no data, and does not answer the donors' sum in its place."""
    _coverage(monkeypatch, official=False)
    part = {
        "entities": [_TO_ETHIOPIA, {"role": "from", "text": "official donors"}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    (answer,) = _resolve(part).parts

    assert isinstance(answer, NoDataPart)
    assert answer.no_data.reason == "no_observations"


_LISTED_OFFICIAL = {"role": "from", "text": "official donors", "members": True}


@pytest.mark.parametrize(
    ("text", "asked", "donors"),
    [
        pytest.param("official donors", {}, _FROM_FRANCE, id="official-donors"),
        pytest.param("all donors", {}, _FROM_FRANCE, id="all-donors"),
        pytest.param("Official Donors, Total", {}, _FROM_FRANCE, id="official-donors-total"),
        pytest.param(
            "official donors", {"members": True}, _FROM_FRANCE, id="official-donors-listed"
        ),
        pytest.param("all donors", {"members": True}, _FROM_FRANCE, id="all-donors-listed"),
        pytest.param(
            "official donors",
            {"operations": [{"over": "origin", "phrase": "number of donors", "verb": "aggregate"}]},
            _FROM_FRANCE,
            id="official-donors-counted",
        ),
        pytest.param(
            "all donors", {}, _LISTED_OFFICIAL, id="all-donors-beside-unlisted-official-donors"
        ),
        pytest.param(
            "all donors",
            {"members": True},
            _LISTED_OFFICIAL,
            id="all-donors-listed-beside-unlisted-official-donors",
        ),
    ],
)
def test_a_whole_population_directed_to_the_other_side_refuses(
    monkeypatch: pytest.MonkeyPatch, text: str, asked: dict[str, Any], donors: dict[str, Any]
):
    """A donor population the request directs to the recipient side ("from France to
    official donors") names parties the code index publishes on the donor side alone, so
    the part refuses `role_incompatible`, and binds neither the population beside France on
    the donor side nor leaves the recipient side unnamed. It refuses so however it is
    asked: posted with `members: true`, or under an operation over the donors, it lists no
    donor side in its place. The conflict is the request's own, so it refuses so before any
    shape is pruned, beside "official donors" posted with `members: true`, which no DAC2a or
    CRS shape lists. The refusal names the text, and only it."""
    _coverage(monkeypatch)
    recipients = {"role": "to", "text": text, "members": asked.get("members", False)}
    part = {
        "entities": [donors, recipients],
        "time": "2021",
        "variable": _VARIABLE,
        **{key: value for key, value in asked.items() if key != "members"},
    }

    (answer,) = _resolve(part).parts

    assert isinstance(answer, NoDataPart)
    assert answer.no_data.reason == "role_incompatible"
    assert answer.no_data.entity_terms == [text]
    assert answer.no_data.message == (
        f'No data for "{_VARIABLE}" as a flow to or from any place named in this query. '
        f'"{text}" belongs on the other side of the flow. Swap its "from" or "to".'
    )


def test_a_total_holding_series_only_for_another_aid_type_leaves_the_side_to_its_parties(
    monkeypatch: pytest.MonkeyPatch,
):
    """Coverage holds the official donors' total to Ethiopia only for another aid type
    than the one the part binds, so the part's own variables hold no series at the total:
    DAC2a answers from its donors, as with no total series at all, and states no total
    beside them."""
    aid_type = "dac2aAidType"
    bound = _DAC2A_MEMBER.split("AidType")[1].split("_")[0]
    other = _DAC2A_MEMBER.replace(f"AidType{bound}_", "AidType207_")
    assert other != _DAC2A_MEMBER and bound != "207"
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (other, "dac2a", _OFFICIAL, _ETH, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
            (_CRS_MEMBER, "crs", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((other, _DAC2A_MEMBER, _CRS_MEMBER), names),
        facet_metadata={
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
        },
    )
    drafted = (
        SlotBindingDraft(
            axis="how",
            property_dcid=aid_type,
            kind="value",
            value_dcids=[f"DAC2aAidType{bound}"],
        ),
    )

    answer = _part(
        _resolve(
            {"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE},
            drafted=drafted,
        )
    )

    assert _origin(answer.spec) is None
    assert _group(answer.spec, "origin") is None
    assert _whole_read(answer) is None
    assert _available(answer) == []


def test_a_declared_default_settles_the_bindings_before_the_places_bind(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bind leaves the aid type open and DAC2a's declared default binds it. Coverage
    holds the official donors' total to Ethiopia only for another aid type, so the bindings
    settled under the total's role keys hold no series at it: the places leave the donor
    side to its parties, and the bindings are settled again under the role keys the answer
    states."""
    bound = _DAC2A_MEMBER.split("AidType")[1].split("_")[0]
    other = _DAC2A_MEMBER.replace(f"AidType{bound}_", "AidType207_")
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (other, "dac2a", _OFFICIAL, _ETH, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
            (_CRS_MEMBER, "crs", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((other, _DAC2A_MEMBER, _CRS_MEMBER), names),
        facet_metadata={
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
        },
    )

    left_open = SlotBindingDraft(
        axis="how", property_dcid="dac2aAidType", kind="unbound", value_dcids=[]
    )
    pinned_under: list[frozenset[str]] = []
    pin = phase_election.canonical_pin_bindings

    def recording_pin(shape: Any, bindings: Any, role_keys: Any) -> Any:
        pinned_under.append(role_keys.bound)
        return pin(shape, bindings, role_keys)

    monkeypatch.setattr(phase_election, "canonical_pin_bindings", recording_pin)

    answer = _part(
        _resolve(
            {"entities": [_TO_ETHIOPIA], "time": "2021", "variable": "aid"},
            drafted=(left_open,),
        )
    )

    assert _origin(answer.spec) is None
    assert _whole_read(answer) is None
    assert pinned_under == [frozenset({"origin", "destination"}), frozenset({"destination"})]


def _never_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """No shape reads any population whole, so no side is offered a total."""
    monkeypatch.setattr(place_roles, "whole_reading", lambda *_args, **_kwargs: None)


def test_a_total_the_measure_vote_leaves_without_series_settles_as_never_offered(
    monkeypatch: pytest.MonkeyPatch,
):
    """Coverage holds the official donors' total to Ethiopia only for DAC2a's net
    disbursements, and the measure vote elects gross ones, a step that runs after the
    places offer the total. The bindings settled under the total's role keys hold no series
    at it, so the donor side is left to its parties, and the answer, bindings included, is
    the one a request offered no total gives, settled under the role keys it states."""
    net = _DAC2A_MEMBER.replace("_Gross_", "_Net_")
    assert net != _DAC2A_MEMBER
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (net, "dac2a", _OFFICIAL, _ETH, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
            (_CRS_MEMBER, "crs", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((net, _DAC2A_MEMBER, _CRS_MEMBER), names),
        facet_metadata={
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
        },
    )
    pinned_under: list[frozenset[str]] = []
    pin = phase_election.canonical_pin_bindings

    def recording_pin(shape: Any, bindings: Any, role_keys: Any) -> Any:
        pinned_under.append(role_keys.bound)
        return pin(shape, bindings, role_keys)

    monkeypatch.setattr(phase_election, "canonical_pin_bindings", recording_pin)
    part = {"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE}

    answer = _part(_resolve(part))
    settled_under = list(pinned_under)
    _never_offered(monkeypatch)
    unoffered = _part(_resolve(part))

    assert _origin(answer.spec) is None
    assert _whole_read(answer) is None
    assert {slot.property: slot.binding for slot in answer.spec.definition.identity.slots}[
        "nettingBasis"
    ].model_dump() == {"state": "value", "ref": "Gross"}
    assert settled_under == [frozenset({"origin", "destination"}), frozenset({"destination"})]
    assert answer.spec.model_dump(mode="json") == unoffered.spec.model_dump(mode="json")
    assert [w.code for w in answer.warnings] == [w.code for w in unoffered.warnings]


def test_the_election_names_its_dataset_from_the_named_places_alone(
    monkeypatch: pytest.MonkeyPatch,
):
    """The election names the dataset the part is served from before any binding is
    settled, so it reads the places the request names and no whole total, which only
    binding repair can judge: Ethiopia alone, though the answer binds the official donors'
    total."""
    _coverage(monkeypatch)
    read_under: list[frozenset[str]] = []
    name = phase_election._name_served_dataset

    def recording_name(*args: Any) -> Any:
        read_under.append(args[-1]().bound)
        return name(*args)

    monkeypatch.setattr(phase_election, "_name_served_dataset", recording_name)

    answer = _part(_resolve({"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE}))

    assert _origin(answer.spec) == [_OFFICIAL]
    assert read_under[0] == frozenset({"destination"})


@pytest.mark.parametrize(
    ("official", "origin"),
    [
        pytest.param(True, [_OFFICIAL], id="total-held"),
        pytest.param(False, None, id="no-total-series"),
    ],
)
def test_a_place_named_without_a_direction_reads_the_total_its_direction_leaves(
    monkeypatch: pytest.MonkeyPatch, official: bool, origin: list[str] | None
):
    """Ethiopia named with no direction holds data only as a recipient, so the answer reads
    it there and states the direction. The donor side that direction leaves unnamed binds
    the official donors' total where coverage holds a series at it, and its donors where
    none."""
    _coverage(monkeypatch, official=official)

    answer = _part(
        _resolve({"entities": [{"text": "Ethiopia"}], "time": "2021", "variable": _VARIABLE})
    )

    spec = answer.spec
    assert spec.definition.identity.entities["destination"] == [_ETH]
    assert _origin(spec) == origin
    assert "direction_inferred" in [departure.code for departure in answer.departures]
    assert (_whole_read(answer) is not None) == official


def test_a_place_without_a_direction_beside_a_named_population_takes_the_other_side(
    monkeypatch: pytest.MonkeyPatch,
):
    """Coverage holds no series, so no reading of Ethiopia, named with no direction, holds
    data and the part refuses `role_not_resolved`. "Official donors" names the official
    donors' total on the donor side, so the refusal asks only Ethiopia's direction and
    offers the one reading that side leaves."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(monkeypatch, [], role_names={_DAC2A_MEMBER: names})
    part = {
        "entities": [{"text": "official donors"}, {"text": "Ethiopia"}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    (answer,) = _resolve(part).parts

    assert isinstance(answer, NoDataPart)
    assert answer.no_data.reason == "role_not_resolved"
    assert answer.no_data.entity_terms == ["Ethiopia"]
    assert [option.label for option in answer.no_data.options or ()] == ["To Ethiopia"]


def _france_coverage(monkeypatch: pytest.MonkeyPatch, *, aid_type: str = "own") -> None:
    """DAC2a holds series in 2021 to France from Germany and from the official donors'
    total, the latter for the official development assistance member's own aid type or for
    `aid_type`, and from France to every recipient and to Ethiopia; the CRS holds France's
    series to Ethiopia."""
    bound = _DAC2A_MEMBER.split("AidType")[1].split("_")[0]
    held = (
        _DAC2A_MEMBER
        if aid_type == "own"
        else _DAC2A_MEMBER.replace(f"AidType{bound}_", f"AidType{aid_type}_")
    )
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (held, "dac2a", _OFFICIAL, _FRA, [2021]),
            (_DAC2A_MEMBER, "dac2a", "country/DEU", _FRA, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ALL_RECIPIENTS, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
            (_CRS_MEMBER, "crs", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((held, _DAC2A_MEMBER, _CRS_MEMBER), names),
        facet_metadata={
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
        },
    )


@pytest.mark.parametrize(
    ("aid_type", "origin"),
    [
        pytest.param("own", [_OFFICIAL], id="total-held-for-its-aid-type"),
        pytest.param("207", None, id="total-held-for-another-aid-type"),
    ],
)
def test_an_alternate_judges_the_total_under_its_own_bindings(
    monkeypatch: pytest.MonkeyPatch, aid_type: str, origin: list[str] | None
):
    """France named with no direction holds data as a donor and as a recipient, so the
    answer reads it as the donor to all recipients and offers the recipient reading beside
    it, an alternate walked under its own representative's bindings. That alternate's
    donor side binds the official donors' total only where coverage holds a series at it
    for the aid type those bindings bind; held for another aid type alone, the side is left
    to its donors, as Germany's series to France serves."""
    _france_coverage(monkeypatch, aid_type=aid_type)

    response = _resolve({"entities": [{"text": "France"}], "time": "2021", "variable": _VARIABLE})

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, CandidatesPart), part
    entities = [spec.definition.identity.entities for spec in part.candidates.specs]
    assert entities == [
        {"destination": [_ALL_RECIPIENTS], "origin": [_FRA]},
        {"destination": [_FRA], **({} if origin is None else {"origin": origin})},
    ]


@pytest.mark.parametrize(
    ("open_role", "listed", "params"),
    [
        pytest.param(
            "donor",
            {"destination": [_FRA]},
            {
                "total": _OFFICIAL,
                "total_label": "official donors (DAC)",
                "whole_of": "official",
            },
            id="donors-listed",
        ),
        pytest.param(
            "recipient",
            {"origin": [_FRA]},
            {
                "total": _ALL_RECIPIENTS,
                "total_label": "all recipients (DAC)",
                "whole_of": "official,residual",
            },
            id="recipients-listed",
        ),
    ],
)
def test_a_candidate_set_states_the_total_beside_the_reading_that_lists_its_side(
    monkeypatch: pytest.MonkeyPatch,
    open_role: str,
    listed: dict[str, list[str]],
    params: dict[str, str],
):
    """France named with no direction offers a reading of each direction. Opening one role
    lists that side member by member in the reading where France stands on the other side,
    and DAC2a publishes that side's total with a series beside France, so the set states
    the total once, stamped with that reading, as a definite answer does."""
    _france_coverage(monkeypatch)

    response = _resolve(
        {
            "entities": [{"text": "France"}],
            "open_roles": [open_role],
            "time": "2021",
            "variable": _VARIABLE,
        }
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, CandidatesPart), part
    (reading,) = [
        spec for spec in part.candidates.specs if spec.definition.identity.entities == listed
    ]
    stated = [
        (warning.spec_id, warning.params)
        for warning in part.warnings
        if warning.code == "whole_total_available"
    ]
    assert stated == [(reading.spec_id, params)]


# DAC2a's gross imputed multilateral aid, a shape of its own beside the official
# development assistance member's, whose representative binds another aid type.
_DAC2A_ATTRIBUTED = "AttributedAmount_FinancialFlow_Gross_Disbursement_DAC2aAidType106_Nominal"


_ETH_PART = {"entities": [_TO_ETHIOPIA], "time": "2021", "variable": "aid"}
_UNNAMED_PART = {"time": "2021", "variable": "aid"}


@pytest.mark.parametrize(
    ("part", "series", "grounded"),
    [
        pytest.param(
            _ETH_PART,
            [(_DAC2A_ATTRIBUTED, _OFFICIAL, _ETH)],
            [
                (_DAC2A_MEMBER, {"destination": [_ETH]}),
                (_DAC2A_ATTRIBUTED, {"destination": [_ETH], "origin": [_OFFICIAL]}),
            ],
            id="total-held-for-the-alternate",
        ),
        pytest.param(
            _ETH_PART,
            [(_DAC2A_MEMBER, _OFFICIAL, _ETH)],
            [
                (_DAC2A_MEMBER, {"destination": [_ETH], "origin": [_OFFICIAL]}),
                (_DAC2A_ATTRIBUTED, {"destination": [_ETH]}),
            ],
            id="total-held-for-the-elected",
        ),
        pytest.param(
            _UNNAMED_PART,
            [(_DAC2A_MEMBER, _OFFICIAL, _FRA), (_DAC2A_ATTRIBUTED, _FRA, _ALL_RECIPIENTS)],
            [
                (_DAC2A_MEMBER, {"origin": [_OFFICIAL]}),
                (_DAC2A_ATTRIBUTED, {"destination": [_ALL_RECIPIENTS]}),
            ],
            id="second-total-held-for-the-alternate",
        ),
    ],
)
def test_an_alternate_of_another_shape_judges_the_total_under_its_own_bindings(
    monkeypatch: pytest.MonkeyPatch,
    part: dict[str, Any],
    series: list[tuple[str, str, str]],
    grounded: list[tuple[str, dict[str, list[str]]]],
):
    """The bind elects DAC2a's official development assistance shape for "aid to Ethiopia",
    binding that aid type, and nominates its imputed multilateral aid shape beside it, whose
    representative binds another. Coverage holds the official donors' total
    to Ethiopia for one of the two variables alone, so the reading whose own bindings elect
    that variable binds the total, and the other leaves its donors unnamed. A part naming
    neither side offers the donors' total, then the recipients' one: where the alternate's
    variable holds no series at the donors' total, it walks on to the recipients' total
    beside the elected reading's donors' total. The answer folds the alternate into the
    elected reading's measure alternatives, so each reading is read as the candidates
    ground it."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            *((variable, "dac2a", origin, dest, [2021]) for variable, origin, dest in series),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
            (_DAC2A_ATTRIBUTED, "dac2a", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), names),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )
    readings: list[tuple[str, dict[str, list[str]]]] = []
    ground = candidate_collapse.ground_candidates

    def recording_ground(*args: Any) -> Any:
        result = ground(*args)
        readings.extend(
            (spec.resolved.members[0].ref.ref, spec.definition.identity.entities)
            for spec in result.specs
        )
        return result

    monkeypatch.setattr(candidate_collapse, "ground_candidates", recording_ground)

    offline_resolve(
        _parsed(part),
        graph=_graph(members=(_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), variable="aid"),
        llm=_ScriptedLLM(
            "aid",
            drafted=(
                SlotBindingDraft(
                    axis="how",
                    property_dcid="dac2aAidType",
                    kind="value",
                    value_dcids=["DAC2aAidType206"],
                ),
            ),
            also_plausible=[1],
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert readings == grounded


def test_a_resubmitted_candidate_set_states_the_total_for_its_own_reading(
    monkeypatch: pytest.MonkeyPatch,
):
    """A candidate set posted back holds DAC2a's donors, whose dataset publishes the
    official donors' total, and the CRS's, which publishes none. The rebuilt set states the
    total once, stamped with DAC2a's reading."""
    _coverage(monkeypatch)
    opened = {
        "entities": [_TO_ETHIOPIA],
        "open_roles": ["donor"],
        "time": "2021",
        "variable": _VARIABLE,
    }
    dac2a = _spec(_resolve(opened))
    crs = _spec(_resolve(opened, crs_first=True))
    assert [source.ref for source in crs.resolved.trace.resolved_sources] == [CRS_PROVENANCE]

    response = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitCandidatesEntry(
                        variable_text=_VARIABLE,
                        ordering="intent_rank",
                        truncated=False,
                        candidates=[
                            ResubmitCandidate(identity=spec.definition.identity)
                            for spec in (dac2a, crs)
                        ],
                    )
                ]
            )
        ),
        graph=_graph(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, CandidatesPart), part
    stated = [
        (warning.spec_id, (warning.params or {}).get("total"))
        for warning in part.warnings
        if warning.code == "whole_total_available"
    ]
    assert stated == [(dac2a.spec_id, _OFFICIAL)]


def test_a_resubmit_refusing_its_posted_aid_type_states_the_total_beside_its_flagged_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """A reading that lists DAC2a's donors to Ethiopia is posted back after the graph has
    given its member another aid type, so the resubmit cannot apply the aid type the
    identity posts and refuses `qualifier_not_bound` with the read without it as its flagged
    match. DAC2a publishes the official donors' total with a series to Ethiopia, so the part
    states the total, stamped with the match, as a fresh request's refusal does."""
    _coverage(monkeypatch)
    opened = {
        "entities": [_TO_ETHIOPIA],
        "open_roles": ["donor"],
        "time": "2021",
        "variable": _VARIABLE,
    }
    posted = _spec(_resolve(opened))
    graph = _graph()
    graph._nodes[_DAC2A_MEMBER]["arcs"]["dac2aAidType"] = {"nodes": [{"dcid": "DAC2aAidType999"}]}

    response = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text=_VARIABLE, identity=posted.definition.identity
                    )
                ]
            )
        ),
        graph=graph,
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert isinstance(response, Resolution), response
    (answer,) = response.parts
    assert isinstance(answer, NoDataPart), answer
    assert answer.no_data.reason == "qualifier_not_bound"
    match = answer.no_data.flagged_match
    assert match is not None
    assert [
        (warning.spec_id, warning.params)
        for warning in answer.warnings
        if warning.code == "whole_total_available"
    ] == [
        (
            match.spec_id,
            {"total": _OFFICIAL, "total_label": "official donors (DAC)", "whole_of": "official"},
        )
    ]


@pytest.mark.parametrize(
    "france",
    [
        pytest.param(_FROM_FRANCE, id="france-directed"),
        pytest.param({"text": "France"}, id="france-undirected"),
    ],
)
def test_a_whole_total_named_beside_another_donor_binds_as_a_named_party(
    monkeypatch: pytest.MonkeyPatch, france: dict[str, str]
):
    """ "France and official donors to Ethiopia" names two donors: the official donors'
    total binds beside France as a party the request names, and no side reads whole."""
    _coverage(monkeypatch)
    part = {
        "entities": [_TO_ETHIOPIA, france, {"role": "from", "text": "official donors"}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    answer = _part(_resolve(part))

    assert [source.ref for source in answer.spec.resolved.trace.resolved_sources] == [_DAC2A]
    assert sorted(_origin(answer.spec) or ()) == [_OFFICIAL, _FRA]
    assert _group(answer.spec, "origin") is None
    assert _whole_read(answer) is None


@pytest.mark.parametrize(
    ("crs_first", "official"),
    [
        pytest.param(True, True, id="crs"),
        pytest.param(False, False, id="dac2a-without-the-total-series"),
    ],
)
def test_a_whole_total_named_beside_another_donor_departs_where_no_series_holds_it(
    monkeypatch: pytest.MonkeyPatch, crs_first: bool, official: bool
):
    """ "Official donors" beside France names a party on any shape that declares the donor
    side. The CRS publishes no total of its providers, and DAC2a holds no series at it
    where coverage records none, so the total binds beside France as named and the answer
    departs `entity_unavailable` for it under its graph label, which the test graph gives as
    its dcid, as for any party with no data."""
    _coverage(monkeypatch, official=official)
    part = {
        "entities": [_TO_ETHIOPIA, _FROM_FRANCE, {"role": "from", "text": "official donors"}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    answer = _part(_resolve(part, crs_first=crs_first))

    spec = answer.spec
    source = CRS_PROVENANCE if crs_first else _DAC2A
    assert [source.ref for source in spec.resolved.trace.resolved_sources] == [source]
    assert sorted(_origin(spec) or ()) == [_OFFICIAL, _FRA]
    assert _group(spec, "origin") is None
    assert _whole_read(answer) is None
    assert [
        (departure.binding_ref, departure.asked)
        for departure in answer.departures
        if departure.code == "entity_unavailable"
    ] == [("origin:", _OFFICIAL)]


_BESIDE_A_NAMED_PLACE = [
    pytest.param(
        [_TO_ETHIOPIA, _FROM_FRANCE, {"role": "from", "text": text}],
        "origin",
        [_OFFICIAL, _PRIVATE, _FRA],
        sources,
        id=f"france-and-{slug}-crs-{how}",
    )
    for text, slug in (
        ("all donors", "all-donors"),
        ("all donors including private", "all-donors-including-private"),
    )
    for sources, how in ((None, "first"), (_CRS_NAMED, "named"))
] + [
    pytest.param(
        [
            _TO_ETHIOPIA,
            _FROM_FRANCE,
            {"role": "from", "text": "official donors"},
            {"role": "from", "text": "all donors"},
        ],
        "origin",
        [_OFFICIAL, _PRIVATE, _FRA],
        None,
        id="france-official-donors-and-all-donors-crs-first",
    ),
    pytest.param(
        [_FROM_FRANCE, _TO_ETHIOPIA, {"role": "to", "text": "all recipients"}],
        "destination",
        [_ALL_RECIPIENTS, _ETH],
        _CRS_NAMED,
        id="ethiopia-and-all-recipients-crs-named",
    ),
]


@pytest.mark.parametrize(("entities", "side", "bound", "sources"), _BESIDE_A_NAMED_PLACE)
def test_a_population_covering_its_side_beside_a_named_place_binds_as_named_parties(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    entities: list[dict[str, str]],
    side: str,
    bound: list[str],
    sources: list[dict[str, str]] | None,
):
    """A population that covers its side on the CRS reads as the side left unnamed only
    where no place is named there. Beside France, "all donors" names its totals as parties,
    and beside Ethiopia so does "all recipients": the CRS holds no series at them, so the
    answer departs `entity_unavailable` for them on that side, as for any named party with
    no data, and states no `whole_total_unpublished`."""
    part: dict[str, Any] = {"entities": entities, "time": "2021", "variable": _VARIABLE}
    if sources is not None:
        part["sources"] = sources
        request.getfixturevalue(source_catalogue.__name__)
    _coverage(monkeypatch)

    answer = _part(_resolve(part, crs_first=True))

    spec = answer.spec
    assert [source.ref for source in spec.resolved.trace.resolved_sources] == [CRS_PROVENANCE]
    assert sorted(spec.definition.identity.entities.get(side) or ()) == bound
    assert [
        departure.binding_ref
        for departure in answer.departures
        if departure.code == "entity_unavailable"
    ] == [role_ref(side)]
    assert "whole_total_unpublished" not in [departure.code for departure in answer.departures]


@pytest.mark.parametrize(
    ("sources", "origin"),
    [
        pytest.param(None, [_OFFICIAL, _PRIVATE], id="crs-first"),
        pytest.param(_CRS_NAMED, None, id="crs-named"),
    ],
)
def test_a_population_beside_one_covering_its_side_binds_as_a_named_party(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    sources: list[dict[str, str]] | None,
    origin: list[str] | None,
):
    """ "Official donors and all donors to Ethiopia" names two populations on the donor
    side. "All donors" covers the CRS's donors and "official donors" does not, so the side
    is named: both bind as named parties, and neither text reads as the side left unnamed.
    The CRS holds no series at either total, so with the CRS searched first DAC2a answers
    with both totals, and with the CRS named the part refuses `no_observations`."""
    part: dict[str, Any] = {
        "entities": [
            _TO_ETHIOPIA,
            {"role": "from", "text": "official donors"},
            {"role": "from", "text": "all donors"},
        ],
        "time": "2021",
        "variable": _VARIABLE,
    }
    if sources is not None:
        part["sources"] = sources
        request.getfixturevalue(source_catalogue.__name__)
    _coverage(monkeypatch)

    (answer,) = _resolve(part, crs_first=True).parts

    if origin is None:
        assert isinstance(answer, NoDataPart), answer
        assert answer.no_data.reason == "no_observations"
        return
    assert isinstance(answer, DefinitePart), answer
    spec = answer.spec
    assert [source.ref for source in spec.resolved.trace.resolved_sources] == [_DAC2A]
    assert _origin(spec) == origin
    assert "whole_total_unpublished" not in [departure.code for departure in answer.departures]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("all donors", id="all-donors"),
        pytest.param("all donors including private", id="all-donors-including-private"),
    ],
)
def test_a_whole_population_asked_member_by_member_lists_its_members(
    monkeypatch: pytest.MonkeyPatch, text: str
):
    """A whole population posted with `members: true` asks for each of its donors, so the
    donor side lists them as `open_roles` does, and no total stands in for them. It lists
    on the CRS, whose donors are official and private ones alone, and answers the Spec
    `open_roles` answers there; DAC2a, offered first, also holds a residual donor ("other
    multilaterals (DAC)") the population's markers leave out."""
    _coverage(monkeypatch)
    part = {
        "entities": [_TO_ETHIOPIA, {"role": "from", "text": text, "members": True}],
        "time": "2021",
        "variable": _VARIABLE,
    }
    opened = {
        "entities": [_TO_ETHIOPIA],
        "open_roles": ["donor"],
        "time": "2021",
        "variable": _VARIABLE,
    }

    answer = _part(_resolve(part))

    spec = answer.spec
    assert _origin(spec) is None
    assert spec.definition.identity.open_roles == ["origin"]
    assert spec.spec_id == _spec(_resolve(opened, crs_first=True)).spec_id
    assert _group(spec, "origin") is None
    assert _whole_read(answer) is None


@pytest.mark.parametrize(
    "asked",
    [
        pytest.param({"members": True}, id="members"),
        pytest.param(
            {
                "operations": [
                    {
                        "over": "origin",
                        "phrase": "sum from official donors giving more than 100 million",
                        "verb": "aggregate",
                    }
                ]
            },
            id="summed-over-a-threshold",
        ),
        pytest.param(
            {
                "operations": [
                    {
                        "over": "origin",
                        "phrase": "number of official donors giving more than 100 million",
                        "verb": "aggregate",
                    }
                ]
            },
            id="counted",
        ),
    ],
)
@pytest.mark.parametrize(
    ("members", "listed", "unread", "held"),
    [
        pytest.param(
            (_COMMITMENTS_MEMBER, _IMPUTED_MEMBER), IMPUTED_PROVENANCE, False, None, id="imputed"
        ),
        pytest.param(
            (_DAC2A_MEMBER, _CRS_MEMBER),
            None,
            False,
            'private donors, which "official donors" leaves out',
            id="dac2a-and-crs",
        ),
        pytest.param(
            (_COMMITMENTS_MEMBER, _IMPUTED_MEMBER),
            None,
            True,
            None,
            id="imputed-donors-unread",
        ),
    ],
)
def test_official_donors_asked_member_by_member_list_where_no_private_donor_stands_beside(
    monkeypatch: pytest.MonkeyPatch,
    members: tuple[str, ...],
    listed: str | None,
    unread: bool,
    held: str | None,
    asked: dict[str, Any],
):
    """ "Official donors" posted with `members: true`, or named under an operation over the
    donors, which may filter or count them, opens the donor side only on a shape whose
    datasets hold official donors alone there, as the imputed multilateral table does. The
    CRS and DAC2a hold private donors beside them, which an open side would list, so a part
    offered those alone refuses `entity_role_not_admitted` for the population, naming the
    classes they hold beside it: DAC2a's private and residual donors, the CRS's private
    ones. Where the index reads no class of donor in the imputed table (`unread`), as for a
    shape no declared dataset publishes, nothing shows what its open side would list, so it
    lists no population either and the part refuses with no account of what the
    population leaves out."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_COMMITMENTS_MEMBER, "crs", _FRA, _ETH, [2021]),
            (_IMPUTED_MEMBER, "imputed", _FRA, _ETH, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
            (_CRS_MEMBER, "crs", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys(members, names),
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD"),
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
        },
    )
    if unread:
        monkeypatch.delitem(get_index().side_classes, ("origin", IMPUTED_PROVENANCE))
    donors = {"role": "from", "text": "official donors", "members": asked.get("members", False)}
    part = {
        "entities": [_TO_ETHIOPIA, donors],
        "time": "2021",
        "variable": "aid",
        **{key: value for key, value in asked.items() if key != "members"},
    }

    (answer,) = offline_resolve(
        _parsed(part),
        graph=_graph(members=members, variable="aid"),
        llm=_ScriptedLLM("aid"),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    ).parts

    if listed is None:
        assert isinstance(answer, NoDataPart), answer
        assert answer.no_data.reason == "entity_role_not_admitted"
        if held is None:
            assert "leaves out" not in answer.no_data.message, answer.no_data.message
            return
        assert answer.no_data.message == (
            '"official donors" cannot be listed donor by donor: every source of the "aid" data '
            f"also holds {held}. Ask for its total, or name the donors to list."
        )
        return
    assert isinstance(answer, DefinitePart), answer
    identity = answer.spec.definition.identity
    assert identity.open_roles == ["origin"]
    assert identity.entities == {"destination": [_ETH]}
    assert [source.ref for source in answer.spec.resolved.trace.resolved_sources] == [listed]


_OFFICIAL_UNLISTED = (
    '"official donors" cannot be listed donor by donor: every source of the '
    f'"{_VARIABLE}" data also holds private donors, which "official donors" leaves out. Ask '
    "for its total, or name the donors to list."
)


@pytest.mark.parametrize(
    "beside",
    [
        pytest.param(
            {"role": "to", "text": "all recipients", "members": True}, id="all-recipients"
        ),
        pytest.param({"role": "from", "text": "all donors", "members": True}, id="all-donors"),
    ],
)
def test_a_population_that_lists_is_left_out_of_the_listing_refusal(
    monkeypatch: pytest.MonkeyPatch, beside: dict[str, Any]
):
    """ "Official donors" posted with `members: true` lists on no DAC2a or CRS shape, since
    both hold private donors, the one class each holds beside it. Beside it, "all recipients"
    lists exactly on both and "all donors" on the CRS, whose donors are official and private
    ones alone, so the refusal names "official donors" alone."""
    _coverage(monkeypatch)
    part = {
        "entities": [
            _LISTED_OFFICIAL,
            beside,
            *([_TO_ETHIOPIA] if beside["role"] == "from" else []),
        ],
        "time": "2021",
        "variable": _VARIABLE,
    }

    (answer,) = _resolve(part).parts

    assert isinstance(answer, NoDataPart), answer
    assert answer.no_data.reason == "entity_role_not_admitted"
    assert answer.no_data.message == _OFFICIAL_UNLISTED


def test_populations_each_listed_alone_refuse_together(monkeypatch: pytest.MonkeyPatch):
    """ "Official donors" and "all recipients", both posted with `members: true`, on the
    imputed multilateral table and the CRS, where the table also holds a class of recipient
    "all recipients" leaves out: "official donors" lists exactly on the table alone and "all
    recipients" on the CRS alone, so no shape lists both and neither is unlistable on its
    own. The refusal names both as ones no source lists together."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    members = (_COMMITMENTS_MEMBER, _IMPUTED_MEMBER)
    patch_coverage(
        monkeypatch,
        [
            (_COMMITMENTS_MEMBER, "crs", _FRA, _ETH, [2021]),
            (_IMPUTED_MEMBER, "imputed", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys(members, names),
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD"),
        },
    )
    monkeypatch.setitem(
        get_index().side_classes,
        ("destination", IMPUTED_PROVENANCE),
        frozenset({"official", "private", "residual"}),
    )
    part = {
        "entities": [
            _LISTED_OFFICIAL,
            {"role": "to", "text": "all recipients", "members": True},
        ],
        "time": "2021",
        "variable": "aid",
    }

    (answer,) = offline_resolve(
        _parsed(part),
        graph=_graph(members=members, variable="aid"),
        llm=_ScriptedLLM("aid"),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    ).parts

    assert isinstance(answer, NoDataPart), answer
    assert answer.no_data.reason == "entity_role_not_admitted"
    assert answer.no_data.message == (
        'No source of the "aid" data can list "official donors" and "all recipients" member '
        "by member together, though each can be listed alone. Search for them separately, or "
        "ask for their totals."
    )


def _index_declared_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have the fake graph answer every code index member's declared roles, as the
    catalogue's arc tables answer them live (`origin_catalogue.graph`), where `_graph` holds
    arcs only for the recalled members. A side opened with no place named elects its role
    class over the whole index family, and a member the fake graph has no arcs for would
    read as declaring no role."""
    rows = {row.dcid: row for row in get_index().statvars}
    recorded = FakeGraph.node_observation_properties_batch

    def declared(graph: FakeGraph, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        held = recorded(graph, dcids)
        return {
            dcid: held.get(dcid)
            or (tuple(rows[dcid].observation_properties) if dcid in rows else ())
            for dcid in dcids
        }

    monkeypatch.setattr(FakeGraph, "node_observation_properties_batch", declared)


@pytest.mark.parametrize(
    "official",
    [pytest.param(True, id="with-official-donors"), pytest.param(False, id="without")],
)
def test_three_populations_each_take_their_own_outcome(
    monkeypatch: pytest.MonkeyPatch, official: bool
):
    """ "All donors" posted with `members: true` lists exactly on the CRS, "official donors"
    posted so lists on no DAC2a or CRS shape, and "all recipients" is named on its own side.
    Beside "official donors" the part refuses `entity_role_not_admitted` for it alone: no
    `role_incompatible`, so "all recipients" is directed to its side, and neither "all
    donors" nor "all recipients" is named. Without it the part answers on the CRS: it lists
    the donors and reads the recipient side as every recipient the CRS holds, departing
    `whole_total_unpublished` for "all recipients" and warning no `role_scope_defaulted`."""
    _coverage(monkeypatch)
    _index_declared_roles(monkeypatch)
    part = {
        "entities": [
            *([_LISTED_OFFICIAL] if official else []),
            {"role": "from", "text": "all donors", "members": True},
            {"role": "to", "text": "all recipients"},
        ],
        "time": "2021",
        "variable": _VARIABLE,
    }

    (answer,) = _resolve(part).parts

    if official:
        assert isinstance(answer, NoDataPart), answer
        assert answer.no_data.reason == "entity_role_not_admitted"
        assert answer.no_data.message == _OFFICIAL_UNLISTED
        return
    assert isinstance(answer, DefinitePart), answer
    spec = answer.spec
    assert [source.ref for source in spec.resolved.trace.resolved_sources] == [CRS_PROVENANCE]
    assert spec.definition.identity.entities == {}
    assert spec.definition.identity.open_roles == ["origin"]
    assert [(departure.code, departure.asked) for departure in answer.departures] == [
        ("whole_total_unpublished", "all recipients")
    ]
    assert [w.code for w in answer.warnings if w.code == "role_scope_defaulted"] == []


@pytest.mark.parametrize(
    ("listed", "named", "side"),
    [
        pytest.param(
            {"role": "to", "text": "all recipients", "members": True},
            _FROM_FRANCE,
            "destination",
            id="all-recipients",
        ),
        pytest.param(
            {"role": "from", "text": "all donors", "members": True},
            _TO_ETHIOPIA,
            "origin",
            id="all-donors",
        ),
    ],
)
def test_a_side_a_listed_population_opens_warns_no_role_scope_defaulted(
    monkeypatch: pytest.MonkeyPatch, listed: dict[str, Any], named: dict[str, Any], side: str
):
    """A whole population posted with `members: true` names its side, so the side it opens
    warns no `role_scope_defaulted` that nothing was named there, where `open_roles` naming
    the same side does."""
    _coverage(monkeypatch)
    part = {"entities": [named, listed], "time": "2021", "variable": _VARIABLE}
    role = "donor" if side == "origin" else "recipient"
    opened = {"entities": [named], "open_roles": [role], "time": "2021", "variable": _VARIABLE}

    def defaulted(part: dict[str, Any]) -> list[str]:
        answer = _part(_resolve(part, crs_first=True))
        assert answer.spec.definition.identity.open_roles == [side]
        return [w.code for w in answer.warnings if w.code == "role_scope_defaulted"]

    assert defaulted(part) == []
    assert defaulted(opened) != []


def test_a_whole_population_asked_member_by_member_beside_a_named_donor_warns(
    monkeypatch: pytest.MonkeyPatch,
):
    """France beside "all donors" posted with `members: true` binds France, which wins over
    the side the population opens, so the part warns that the breakdown over every donor was
    not applied, as for France beside `open_roles: ["donor"]`."""
    _coverage(monkeypatch)
    listed = {
        "entities": [
            _TO_ETHIOPIA,
            {"text": "France"},
            {"role": "from", "text": "all donors", "members": True},
        ],
        "time": "2021",
        "variable": _VARIABLE,
    }
    opened = {
        "entities": [_TO_ETHIOPIA, {"text": "France"}],
        "open_roles": ["donor"],
        "time": "2021",
        "variable": _VARIABLE,
    }

    def dropped(part: dict[str, Any]) -> list[str]:
        answer = _part(_resolve(part))
        assert _origin(answer.spec) == [_FRA]
        return [w.message for w in answer.warnings if w.code == "query_qualifier_dropped"]

    assert dropped(listed) == dropped(opened) != []


_OFFICIAL_SHOWN = (
    "official donors (DAC), the source's total for every official donor. Private donors are "
    "outside it."
)


@pytest.mark.parametrize(
    ("part", "text"),
    [
        pytest.param(
            {"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE},
            f"No donor was named, so this shows {_OFFICIAL_SHOWN} To list each donor, search "
            "again with open_roles ['donor'].",
            id="no-donor-named",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "all donors"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "You asked for all donors. This shows official donors (DAC) plus private donors "
            "(DAC). To list each donor, search again with open_roles ['donor'] in place of all "
            "donors.",
            id="all-donors-named",
        ),
        pytest.param(
            {
                "entities": [
                    _TO_ETHIOPIA,
                    {"role": "from", "text": "all donors including private"},
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "You asked for all donors including private. This shows official donors (DAC) plus "
            "private donors (DAC). To list each donor, search again with open_roles ['donor'] "
            "in place of all donors including private.",
            id="all-donors-including-private",
        ),
        pytest.param(
            {"entities": [_FROM_FRANCE], "time": "2021", "variable": _VARIABLE},
            "No recipient was named, so this shows all recipients (DAC), the source's total for "
            "every recipient, which includes aid not allocated to a country. To list each "
            "recipient, search again with open_roles ['recipient'].",
            id="no-recipient-named",
        ),
    ],
)
def test_a_side_read_whole_departs_with_the_way_back_to_its_members(
    monkeypatch: pytest.MonkeyPatch, part: dict[str, Any], text: str
):
    """The `whole_total_read` departure says what the side's total holds and how to list
    its members: the request's words for the population are replaced by `open_roles`."""
    _coverage(monkeypatch)

    (departure,) = [
        departure
        for departure in _part(_resolve(part)).departures
        if departure.code == "whole_total_read"
    ]

    assert departure.text == text


_EXTRACTED_ETHIOPIA = ExtractedEntity(
    text="Ethiopia", span="Ethiopia", flow_role="destination", role_span="to"
)


def _raw(
    query: str,
    *,
    dropped: tuple[str, ...] = (),
    operations: tuple[ExtractedOperation, ...] = (),
    entities: tuple[ExtractedEntity, ...] = (),
) -> DefinitePart:
    """The one part the raw-text door answers `query` with, extraction reading the
    variable, Ethiopia as recipient and `entities`, and dropping `dropped`."""
    return _part(
        offline_resolve(
            ResolveRequest(input=RawTextInput(query=query)),
            graph=_graph(),
            llm=_ScriptedLLM(
                _VARIABLE,
                dropped=dropped,
                operations=operations,
                entities=(_EXTRACTED_ETHIOPIA, *entities),
            ),
            sdmx=FakeSdmxClient(observations={}, availability={}),
        )
    )


@pytest.mark.parametrize(
    ("query", "extracted"),
    [
        pytest.param(
            "official development assistance to Ethiopia by donor",
            {
                "operations": (
                    ExtractedOperation(verb="rank", variable=0, phrase="by donor", over="origin"),
                )
            },
            id="by-donor-read-as-a-rank",
        ),
        pytest.param(
            "official development assistance to Ethiopia by donor",
            {"dropped": ("by donor",)},
            id="by-donor-dropped",
        ),
        pytest.param(
            "official development assistance to Ethiopia for each donor",
            {
                "operations": (
                    ExtractedOperation(
                        verb="aggregate", variable=0, phrase="for each donor", over="origin"
                    ),
                )
            },
            id="each-donor-read-as-a-sum",
        ),
        pytest.param(
            "official development assistance to Ethiopia from each donor",
            {"dropped": ("from each donor",)},
            id="each-donor-dropped",
        ),
        pytest.param(
            "official development assistance to Ethiopia from every donor",
            {
                "entities": (
                    ExtractedEntity(
                        text="every donor",
                        span="every donor",
                        flow_role="origin",
                        role_span="from",
                    ),
                )
            },
            id="every-donor-read-as-a-place",
        ),
    ],
)
def test_a_raw_text_breakdown_by_a_role_lists_its_members(
    monkeypatch: pytest.MonkeyPatch, query: str, extracted: dict[str, Any]
):
    """On the raw-text door, "by donor", "each donor" and "every donor" ask for one value
    per donor, however extraction carries them: the part opens the donor side as
    `open_roles: ["donor"]` does on the parsed door, reads no whole total, runs no
    operation for the phrase and drops no qualifier."""
    _coverage(monkeypatch)

    answer = _raw(query, **extracted)

    spec = answer.spec
    assert spec.definition.identity.open_roles == ["origin"]
    assert _origin(spec) is None
    assert _whole_read(answer) is None
    assert [d.code for d in answer.departures if d.code == "operation_not_applied"] == []
    assert [w.code for w in answer.warnings if w.code == "query_qualifier_dropped"] == []


@pytest.mark.parametrize(
    ("query", "phrase"),
    [
        pytest.param(
            "total official development assistance to Ethiopia across donors",
            "across donors",
            id="sum-across-donors",
        ),
        pytest.param(
            "official development assistance to Ethiopia by donor",
            "by donor",
            id="sum-by-donor",
        ),
    ],
)
def test_a_raw_text_sum_tagged_with_breakdown_words_keeps_its_operation(
    monkeypatch: pytest.MonkeyPatch, query: str, phrase: str
):
    """ "By" or "across" a role reads as a breakdown only on a rank. A sum tagged with
    either keeps its operation on the part and departs as not applied, as any operation
    Search does not run."""
    _coverage(monkeypatch)

    answer = _raw(
        query,
        operations=(
            ExtractedOperation(verb="aggregate", variable=0, phrase=phrase, over="origin"),
        ),
    )

    assert [(op.verb, op.phrase) for op in answer.operations] == [("aggregate", phrase)]
    assert [
        departure.asked
        for departure in answer.departures
        if departure.code == "operation_not_applied"
    ] == [phrase]


def test_a_raw_text_side_read_whole_names_the_words_that_list_its_members(
    monkeypatch: pytest.MonkeyPatch,
):
    """The raw-text door cannot post `open_roles`, so a donor side read whole names the
    words that list its donors, which the door reads as the breakdown."""
    _coverage(monkeypatch)

    answer = _raw("official development assistance to Ethiopia")

    (departure,) = [d for d in answer.departures if d.code == "whole_total_read"]
    assert departure.text.endswith('To list each donor, add "by donor" to the question.')


# DAC2b's net other official flows member, a family DAC2a and the CRS do not publish, with
# the bindings that settle it to one StatVar. "Other official flows" names its aid type.
_DAC2B_MEMBER = "Amount_FinancialFlow_Net_Disbursement_DAC2bAidType972_Nominal"
_DAC2B_DRAFTED = tuple(
    SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
    for prop, value in (
        ("dac2bAidType", "DAC2bAidType972"),
        ("financialFlowType", "Disbursement"),
        ("nettingBasis", "Net"),
    )
)


def _dac2b_part(
    monkeypatch: pytest.MonkeyPatch, part: dict[str, Any], *, origin: str, destination: str
) -> DefinitePart:
    """The one part answering `part` from DAC2b's member, whose coverage holds a series
    from `origin` to `destination` in 2020."""
    patch_coverage(
        monkeypatch,
        [(_DAC2B_MEMBER, "dac2b", origin, destination, [2020])],
        role_names={
            _DAC2B_MEMBER: RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
        facet_metadata={"dac2b": FacetMetadata(provenance_id=_DAC2B, unit="USD")},
    )
    response = offline_resolve(
        _parsed(part),
        graph=_graph(members=(_DAC2B_MEMBER,), variable=part["variable"]),
        llm=_ScriptedLLM(part["variable"], drafted=_DAC2B_DRAFTED),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    return _part(response)


def test_a_total_dac2b_publishes_unmarked_binds_as_a_named_party(
    monkeypatch: pytest.MonkeyPatch,
):
    """DAC2b publishes "All Recipients, Total" with no whole marker, so "other official
    flows from France to all recipients" binds it as a named party there: the recipient side
    is not left unnamed as on a dataset publishing no recipients' total, and reads no whole
    population."""
    part = {
        "entities": [_FROM_FRANCE, {"role": "to", "text": "all recipients"}],
        "time": "2020",
        "variable": "other official flows",
    }

    answer = _dac2b_part(monkeypatch, part, origin=_FRA, destination=_ALL_RECIPIENTS)

    spec = answer.spec
    assert spec.definition.identity.entities == {
        "destination": [_ALL_RECIPIENTS],
        "origin": [_FRA],
    }
    assert _group(spec, "destination") is None
    assert [
        departure.code
        for departure in answer.departures
        if departure.code in ("whole_total_read", "whole_total_unpublished")
    ] == []


def test_the_official_donors_total_names_private_donors_where_its_dataset_holds_them(
    monkeypatch: pytest.MonkeyPatch,
):
    """DAC2a and DAC2b both mark "Official Donors, Total" whole. DAC2a publishes private
    donors beside it, so the role's group lists `private` among its `side_classes` and its
    `whole_total_read` says they are outside the total; DAC2b publishes none, so its group
    lists no `private` and its departure says nothing of them."""
    _coverage(monkeypatch)
    dac2a = _part(_resolve({"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE}))
    dac2b = _dac2b_part(
        monkeypatch,
        {"entities": [_TO_ETHIOPIA], "time": "2020", "variable": "other official flows"},
        origin=_OFFICIAL,
        destination=_ETH,
    )

    def read(answer: DefinitePart) -> str:
        (text,) = [d.text for d in answer.departures if d.code == "whole_total_read"]
        return text

    assert _group(dac2a.spec, "origin") == _OFFICIAL_GROUP
    assert "Private donors are outside it." in read(dac2a)
    assert _group(dac2b.spec, "origin") == {
        **_OFFICIAL_GROUP,
        "side_classes": ["official", "residual"],
    }
    assert "Private donors" not in read(dac2b)


@pytest.mark.parametrize(
    "posted",
    [
        pytest.param(None, id="no-group-posted"),
        pytest.param(
            {
                "origin": EntityGroup.model_validate(
                    {**_OFFICIAL_GROUP, "side_classes": ["official"]}
                )
            },
            id="side-classes-posted",
        ),
    ],
)
def test_a_resubmit_restores_the_whole_population_from_the_identity(
    monkeypatch: pytest.MonkeyPatch, posted: dict[str, EntityGroup] | None
):
    """A resubmit of the answer's identity reads the official donors' total as the donor
    side's whole population again, and answers the same Spec, with the classes DAC2a
    publishes on the donor side whatever group it posts. Its door holds no words of the
    request, so the departure states only what the answer shows."""
    _coverage(monkeypatch)
    first = _spec(_resolve({"entities": [_TO_ETHIOPIA], "time": "2021", "variable": _VARIABLE}))

    response = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text=_VARIABLE,
                        identity=first.definition.identity,
                        entity_groups=posted,
                    )
                ]
            )
        ),
        graph=_graph(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    answer = _part(response)
    assert answer.spec.spec_id == first.spec_id
    assert _group(answer.spec, "origin") == _OFFICIAL_GROUP
    (departure,) = [d for d in answer.departures if d.code == "whole_total_read"]
    assert departure.asked is None
    assert departure.text.startswith(f"This shows {_OFFICIAL_SHOWN}")


@pytest.mark.parametrize(
    ("dcids", "datasets", "years", "reading"),
    [
        pytest.param([_OFFICIAL], [_DAC2A], None, ("origin", ("official",)), id="official-dac2a"),
        pytest.param([_OFFICIAL], [_DAC2B], None, ("origin", ("official",)), id="official-dac2b"),
        pytest.param([_OFFICIAL], [CRS_PROVENANCE], None, None, id="official-crs"),
        pytest.param([_PRIVATE], [_DAC2A], None, None, id="private-alone"),
        pytest.param(
            [_PRIVATE, _OFFICIAL], [_DAC2A], None, ("origin", ("official", "private")), id="set"
        ),
        pytest.param([_PRIVATE, _OFFICIAL], [_DAC2B], None, None, id="set-dac2b"),
        pytest.param(
            [_ALL_RECIPIENTS],
            [_DAC2A],
            (2021, 2021),
            ("destination", ("official", "residual")),
            id="recipients-2021",
        ),
        pytest.param([_ALL_RECIPIENTS], [_DAC2A], (1983, 1983), None, id="recipients-1983"),
        pytest.param([_ALL_RECIPIENTS], [_DAC2A], None, None, id="recipients-no-period"),
        pytest.param([_ALL_RECIPIENTS], [_DAC2B], (2021, 2021), None, id="recipients-dac2b"),
        pytest.param([_ALL_RECIPIENTS], [], (2021, 2021), None, id="no-dataset"),
    ],
)
def test_a_side_reads_whole_where_every_dataset_marks_its_parties(dcids, datasets, years, reading):
    """A side reads as its whole population only where each dataset marks its parties
    within the asked years and the population covers the official parties: DAC2b marks the
    official donors' total and not the recipients', and the recipients' total holds only
    from 1984 in DAC2a."""
    result = whole_reading(dcids, datasets=datasets, years=years)

    assert (None if result is None else (result.axis, result.whole_of)) == reading


@pytest.mark.parametrize(
    ("dcids", "datasets", "covers"),
    [
        pytest.param([_ALL_RECIPIENTS], [CRS_PROVENANCE], True, id="crs-recipients"),
        pytest.param([_OFFICIAL, _PRIVATE], [CRS_PROVENANCE], True, id="crs-all-donors"),
        pytest.param([_OFFICIAL], [CRS_PROVENANCE], False, id="crs-official-donors"),
        pytest.param([_OFFICIAL], [_DAC1], True, id="dac1-official-donors"),
        pytest.param([_ALL_RECIPIENTS], [_DAC2A], False, id="dac2a-marks-the-recipients"),
        pytest.param(
            [_ALL_RECIPIENTS], [_DAC2B], False, id="dac2b-publishes-the-recipients-unmarked"
        ),
        pytest.param([_ALL_RECIPIENTS], ["provenance/Unindexed"], False, id="no-party-held"),
        pytest.param([_ALL_RECIPIENTS], [], False, id="no-dataset"),
    ],
)
def test_a_population_covers_a_side_its_datasets_publish_no_total_of(dcids, datasets, covers):
    """A whole population reads as its side left unnamed where none of the datasets publishes
    it and every party they hold there, totals aside, is of a class it covers. The CRS's
    recipients are all countries or residuals, its donors official or private, and DAC1's
    donors official alone; DAC2a marks the recipients' total, and DAC2b publishes it
    unmarked; a dataset holding no party on the side covers nothing."""
    assert covers_unpublished(dcids, datasets=datasets) is covers


_RECIPIENTS_GROUP = {
    "label": "all recipients (DAC)",
    "side_classes": ["official", "residual"],
    "total": _ALL_RECIPIENTS,
    "whole_of": ["official", "residual"],
}
_RECIPIENTS_AVAILABLE = {
    "total": _ALL_RECIPIENTS,
    "total_label": "all recipients (DAC)",
    "whole_of": "official,residual",
}


@pytest.mark.parametrize(
    ("part", "whole", "destination", "listed"),
    [
        pytest.param(
            {
                "entities": [_FROM_FRANCE, {"role": "to", "text": "all recipients"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            True,
            [_ALL_RECIPIENTS],
            [],
            id="all-recipients-named",
        ),
        pytest.param(
            {"entities": [_FROM_FRANCE], "time": "2021", "variable": _VARIABLE},
            True,
            [_ALL_RECIPIENTS],
            [],
            id="no-recipient-named",
        ),
        pytest.param(
            {
                "entities": [_FROM_FRANCE, {"role": "to", "text": "all recipients"}],
                "time": "1983",
                "variable": _VARIABLE,
            },
            False,
            [_ALL_RECIPIENTS],
            [],
            id="all-recipients-named-before-the-window",
        ),
        pytest.param(
            {"entities": [_FROM_FRANCE], "variable": _VARIABLE},
            False,
            None,
            [],
            id="no-period",
        ),
        pytest.param(
            {
                "entities": [_FROM_FRANCE],
                "operations": [
                    {"over": "destination", "phrase": "top 10 recipients", "verb": "rank"}
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            False,
            None,
            [_RECIPIENTS_AVAILABLE],
            id="recipients-ranked",
        ),
    ],
)
def test_a_recipient_side_reads_whole_inside_the_window(
    monkeypatch: pytest.MonkeyPatch,
    part: dict[str, Any],
    destination: list[str] | None,
    whole: bool,
    listed: list[dict[str, str]],
):
    """DAC2a's all-recipients total holds every recipient from 1984 to 2024: France's aid
    to all recipients, or to no recipient named, in 2021 reads it whole. Named in 1983, it
    binds as a named party outside the window, with no whole reading. To no recipient named
    over no stated period, or ranked over the recipients, the part reads the recipients.
    Only the ranked recipients, listed in 2021, are listed inside the total's window, so
    only they state it beside them."""
    _coverage(monkeypatch, years=(1983, 2021))

    answer = _part(_resolve(part))

    spec = answer.spec
    assert spec.definition.identity.entities.get("destination") == destination
    assert _origin(spec) == [_FRA]
    assert _group(spec, "destination") == (_RECIPIENTS_GROUP if whole else None)
    assert (_whole_read(answer) is not None) == whole
    assert _available(answer) == listed


@pytest.mark.parametrize(
    ("official", "entities"),
    [
        pytest.param(True, {"origin": [_OFFICIAL]}, id="donors-total-held"),
        pytest.param(False, {"destination": [_ALL_RECIPIENTS]}, id="no-donors-total-series"),
    ],
)
def test_both_sides_unnamed_bind_one_total(
    monkeypatch: pytest.MonkeyPatch, official: bool, entities: dict[str, list[str]]
):
    """A part naming neither side binds the official donors' total and leaves the
    recipients unnamed: the source publishes no figure from every donor to every recipient,
    so the recipients' total is not stated beside them either. Where no series holds the
    donors' total, the part binds the recipients' total in its place, which France's series
    to all recipients holds, and states no total beside it."""
    _coverage(monkeypatch, official=official)

    answer = _part(_resolve({"time": "2021", "variable": _VARIABLE}))

    assert answer.spec.definition.identity.entities == entities
    assert _available(answer) == []


def test_a_named_whole_population_leaves_the_other_side_without_a_total(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "ODA from official donors in 2021" names the donors whole and no recipient. The
    recipients are totalled, but the source publishes no figure between two whole totals,
    so the part offers no recipients' total beside the named donors even where coverage
    holds a series between the two: the donor side alone reads whole."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_DAC2A_MEMBER, "dac2a", _OFFICIAL, _ALL_RECIPIENTS, [2021]),
            (_DAC2A_MEMBER, "dac2a", _OFFICIAL, _ETH, [2021]),
        ],
        role_names={_DAC2A_MEMBER: names},
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )
    part = {
        "entities": [{"role": "from", "text": "official donors"}],
        "time": "2021",
        "variable": _VARIABLE,
    }

    answer = _part(_resolve(part))

    spec = answer.spec
    assert spec.definition.identity.entities == {"origin": [_OFFICIAL]}
    assert _group(spec, "origin") == _OFFICIAL_GROUP
    assert _group(spec, "destination") is None
    assert _whole_read(answer) == ("origin:", "official donors")


def test_a_named_whole_population_asks_for_its_side():
    """ "France to all recipients" names the recipient side as a whole population, so a
    shape must declare that side, as it must for a named recipient: a donor-only table
    cannot answer it."""
    france = PlaceSet(name="France", kind="place", node=PlaceNode(dcid=_FRA, graph_type="Country"))
    state = SimpleNamespace(
        entities=["France", "all recipients"],
        place_sets={"France": france},
        whole_named={"all recipients": (_ALL_RECIPIENTS,)},
        entity_roles={
            "France": [DoorRole(axis="origin", span="from")],
            "all recipients": [DoorRole(axis="destination", span="to")],
        },
        operations=(),
    )

    assert query_role_axes(cast(ResolveState, state)) == {"origin", "destination"}


def _answers_for(response: ResolveResponse, text: str, side: str) -> bool:
    """Whether the one part of `response` answers for the request's `text` on `side`: it
    refuses, binds a party on `side`, or departs asking `text`."""
    (part,) = response.parts
    if not isinstance(part, DefinitePart):
        return True
    if part.spec.definition.identity.entities.get(side):
        return True
    return any(departure.asked == text for departure in part.departures)


_ALL_RECIPIENTS_NAMED = [_FROM_FRANCE, {"role": "to", "text": "all recipients"}]


@pytest.mark.parametrize(
    ("part", "text", "side", "crs_first"),
    [
        pytest.param(
            {"entities": _ALL_RECIPIENTS_NAMED, "variable": _VARIABLE},
            "all recipients",
            "destination",
            False,
            id="all-recipients-no-period",
        ),
        pytest.param(
            {"entities": _ALL_RECIPIENTS_NAMED, "time": "1983", "variable": _VARIABLE},
            "all recipients",
            "destination",
            False,
            id="all-recipients-before-the-window",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "all donors"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "all donors",
            "origin",
            True,
            id="crs-first-all-donors",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "official donors"}],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "official donors",
            "origin",
            True,
            id="crs-first-official-donors",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "official donors"}],
                "sources": _CRS_NAMED,
                "time": "2021",
                "variable": _VARIABLE,
            },
            "official donors",
            "origin",
            True,
            id="crs-named-official-donors",
        ),
        pytest.param(
            {
                "entities": [_TO_ETHIOPIA, {"role": "from", "text": "all donors"}],
                "sources": _CRS_NAMED,
                "time": "2021",
                "variable": _VARIABLE,
            },
            "all donors",
            "origin",
            True,
            id="crs-named-all-donors",
        ),
        pytest.param(
            {
                "entities": [
                    _TO_ETHIOPIA,
                    {"role": "from", "text": "all donors including private"},
                ],
                "time": "2021",
                "variable": _VARIABLE,
            },
            "all donors including private",
            "origin",
            True,
            id="crs-first-all-donors-including-private",
        ),
        pytest.param(
            {
                "entities": _ALL_RECIPIENTS_NAMED,
                "sources": _CRS_NAMED,
                "time": "2021",
                "variable": _VARIABLE,
            },
            "all recipients",
            "destination",
            True,
            id="crs-named-all-recipients",
        ),
    ],
)
def test_a_whole_population_no_total_reads_still_answers_for_its_text(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    part: dict[str, Any],
    text: str,
    side: str,
    crs_first: bool,
):
    """A whole population named where the elected shape offers no total for it binds a
    party on its side, departs asking for it, or refuses. DAC2a's all-recipients total
    holds from 1984, so France's aid to all recipients over no period or in 1983 reads no
    total; the CRS marks no donor total, so "all donors" and "official donors" read none
    there, whether the CRS is searched first or named as the source. "All donors including
    private" and "all recipients" cover the CRS's donors and recipients, so the CRS reads
    their side as unnamed, and says so."""
    if "sources" in part:
        request.getfixturevalue(source_catalogue.__name__)
    _coverage(monkeypatch, years=(1983, 2021))

    response = _resolve(part, crs_first=crs_first)

    assert _answers_for(response, text, side)


def test_an_alternate_is_walked_once_under_the_assignment_judged_before_its_walk(
    monkeypatch: pytest.MonkeyPatch,
):
    """A part naming neither side offers the imputed multilateral alternate the donors'
    total and the recipients' total, and coverage holds neither for the alternate's
    variable. The assignment it is walked under is judged once, under the bindings
    `candidates.settle_alternate` gives it before the walk, and the alternate is walked
    once, under that assignment: a member walk of the mint's graph is a few graph round
    trips, and none runs for a total the judgement left out."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_DAC2A_MEMBER, "dac2a", _OFFICIAL, _FRA, [2021]),
            (_DAC2A_MEMBER, "dac2a", _FRA, _ETH, [2021]),
            (_DAC2A_ATTRIBUTED, "dac2a", _FRA, _ETH, [2021]),
        ],
        role_names=dict.fromkeys((_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), names),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )
    judged: list[str] = []
    walked: list[frozenset[str]] = []
    judge = phase_place_materialise._walk_assignment
    walk = phase_place_materialise.walk_alternate

    def recording_judge(state: Any, index: int, shape: Any, rep_sv: str) -> Any:
        if shape.meas_prop_dcid == "attributedAmount":
            judged.append(rep_sv)
        return judge(state, index, shape, rep_sv)

    def recording_walk(**kwargs: Any) -> Any:
        if kwargs["shape"].meas_prop_dcid == "attributedAmount":
            walked.append(frozenset(kwargs["role_keys"].bound))
        return walk(**kwargs)

    monkeypatch.setattr(phase_place_materialise, "_walk_assignment", recording_judge)
    monkeypatch.setattr(phase_place_materialise, "walk_alternate", recording_walk)
    graph = _graph(members=(_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), variable="aid")

    offline_resolve(
        _parsed(_UNNAMED_PART),
        graph=graph,
        llm=_ScriptedLLM(
            "aid",
            drafted=(
                SlotBindingDraft(
                    axis="how",
                    property_dcid="dac2aAidType",
                    kind="value",
                    value_dcids=["DAC2aAidType206"],
                ),
            ),
            also_plausible=[1],
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=member_walk_reader(graph),
    )

    assert len(judged) == 1, judged
    assert walked == [frozenset()], walked


def test_naming_the_served_dataset_leaves_the_declared_datasets_as_they_were():
    """Binding repair binds the query's places (`place_roles.bind_place_roles`) and offers
    whole totals (`place_roles.whole_total_candidates`) on the elected shape before it
    names the dataset the part is served from (`phase_election._served`), and both read
    the shape's declared datasets (`bound_values.declared_datasets`). Naming picks one of
    the datasets publishing the shape's family, so it leaves them unchanged only while
    every family of the packaged index has one publisher."""
    families = [family for found in get_index().families_by_five_tuple.values() for family in found]

    assert families
    assert [family.datasets for family in families if len(family.datasets) != 1] == []
