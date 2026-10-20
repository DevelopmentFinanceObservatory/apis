"""Checks what a resolved Spec assumed (`render.assumptions`) on Specs the sentence panel
recorded live, and the clause a candidates part's own sentence closes with.

The panel keeps each Spec's trace slot filters, so a declared default the election fixed
reads as recorded; a test that needs one on another slot records it there as the election
does (`provenance="declared_default"`).
"""

from __future__ import annotations

import gzip
import json
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.sentence_golden import golden_parts
from search.models import (
    Assumption,
    BreadthDim,
    CandidatesPart,
    CoverageBreadth,
    DefinitePart,
    ResolvedEntity,
    ResolvedFilter,
    ResolvedScope,
    Spec,
)
from search.render.assumptions import assumptions_for, common_assumptions
from search.render.candidates import candidates_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import _spec, _what_slot

_PANEL = Path(__file__).parent.parent / "fixtures" / "sentence_panel"

_EVERY_PROVIDER = Assumption(kind="open_role", binding_ref="origin:", text="every provider")
_EVERY_RECIPIENT = Assumption(kind="open_role", binding_ref="where:", text="every recipient")
_EVERY_CREDITOR = Assumption(kind="open_role", binding_ref="origin:", text="every creditor")


@cache
def _panel_part(query_id: str) -> DefinitePart | CandidatesPart:
    """The first part of the panel line `query_id`, each Spec stating its sentence facts."""
    path = _PANEL / f"{query_id.split('-')[0]}.jsonl.gz"
    for text in gzip.decompress(path.read_bytes()).splitlines():
        line = json.loads(text)
        if line["id"] == query_id:
            part = golden_parts(line["parts"])[0]
            assert isinstance(part, DefinitePart | CandidatesPart)
            return part
    raise KeyError(query_id)


def _panel_spec(query_id: str) -> Spec:
    part = _panel_part(query_id)
    assert isinstance(part, DefinitePart)
    return part.spec


def _open_roles(assumptions: list[Assumption]) -> list[Assumption]:
    return [item for item in assumptions if item.kind == "open_role"]


def _with_declared_default(spec: Spec, ref: str) -> Spec:
    """`spec` with its slot on `ref` recorded as the election's declared default."""
    slot = next(s for s in spec.resolved.slots if s.key.as_ref() == ref)
    binding = slot.binding
    refs = [binding.value.ref] if binding.kind == "value" and binding.value.ref else []
    default = ResolvedFilter(
        key=slot.key, binding_kind="value", refs=refs, provenance="declared_default"
    )
    trace = spec.resolved.trace.model_copy(update={"slot_filters": [default]})
    return spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"trace": trace})})


@pytest.mark.parametrize(
    ("query_id", "binding_ref", "ref"),
    [
        pytest.param("crs-002", "how:dac2aAidType", "DAC2aAidType206", id="crs-002 DAC2a"),
        # The CRS flow code's short name is "ODA"; the clause states its full label.
        pytest.param("crs-026", "how:crsTypeOfFlow", "CRSTypeOfFlow10", id="crs-026 CRS"),
    ],
)
def test_aid_to_a_named_recipient_assumes_the_default_flow_and_every_provider(
    query_id: str, binding_ref: str, ref: str
):
    """crs-002, "aid to Bangladesh", and crs-026, malaria control in Uganda: the flow the
    recorded declared default fixed, named by the code's full label, then the providers no
    word named."""
    assert assumptions_for(_panel_spec(query_id)) == [
        Assumption(
            kind="declared_default",
            binding_ref=binding_ref,
            text="official development assistance",
            refs=[ref],
        ),
        _EVERY_PROVIDER,
    ]


def test_a_declared_default_the_heading_states_takes_the_code_label():
    """crs-026: a purpose slot the heading states ("for malaria control") is named by its
    code's label, as a slot the heading leaves silent is."""
    spec = _with_declared_default(_panel_spec("crs-026"), "how:crsPurpose")
    assert assumptions_for(spec)[0] == Assumption(
        kind="declared_default",
        binding_ref="how:crsPurpose",
        text="malaria control",
        refs=["CRSPurposeCode12262"],
    )


@pytest.mark.parametrize(
    ("query_id", "expected"),
    [
        pytest.param("crs-001", [], id="crs-001 both roles named"),
        # DAC1 declares no recipient, so "worldwide" reads across every provider alone.
        pytest.param("crs-004", [_EVERY_PROVIDER], id="crs-004 none named"),
        pytest.param("mls-004", [_EVERY_RECIPIENT, _EVERY_PROVIDER], id="mls-004 none named"),
        pytest.param("ids-001", [_EVERY_CREDITOR], id="ids-001 debt service of Ghana"),
        pytest.param("ids-008", [], id="ids-008 owed to bondholders"),
        pytest.param("weo-001", [], id="weo-001 subject place named"),
    ],
)
def test_open_roles_are_the_roles_the_heading_states_open(
    query_id: str, expected: list[Assumption]
):
    """An open role is one the heading counts or reads as all ("owed to all creditors") on
    an axis the identity neither binds nor opens, named by its family's noun. A role a how
    value narrows ("owed to bondholders") states none, and a fully asked question lists
    none."""
    assert _open_roles(assumptions_for(_panel_spec(query_id))) == expected


def test_a_role_asked_member_by_member_is_no_assumption():
    """A role in open_roles, as "top 5 donors" opens it, was asked for."""
    spec = _panel_spec("crs-002")
    identity = spec.definition.identity.model_copy(update={"open_roles": ["origin"]})
    definition = spec.definition.model_copy(update={"identity": identity})
    assert _open_roles(assumptions_for(spec.model_copy(update={"definition": definition}))) == []


@pytest.mark.parametrize(
    ("label", "members", "text"),
    [
        ("places", [("country/KEN", "Country"), ("geoId/06", "State")], "every place"),
        ("places", [("country/GHA", "Country"), ("country/KEN", "Country")], "every country"),
        ("countries", [("country/KEN", "Country"), ("country/UGA", None)], "every place"),
        ("worldwide", [("Earth", "Place")], "the world as a whole"),
    ],
)
def test_a_request_naming_no_place_assumes_the_place_scope_as_its_members_are_typed(
    label: str, members: list[tuple[str, str | None]], text: str
):
    """ "government debt in 2020" reads across 46 places: "every country" when every member
    is a Country, whatever the coverage dimension counts, else "every place"."""
    spec = _spec(
        slots=[_what_slot("Gross public sector debt")],
        coverage=CoverageBreadth(
            has_data=True, window=None, dimensions=[BreadthDim(label=label, count=len(members))]
        ),
        scope=ResolvedScope(
            axis="observationAbout",
            members=[
                ResolvedEntity(ref=ref, role="observationAbout", entity_type=entity_type)
                for ref, entity_type in members
            ],
        ),
    )
    assert assumptions_for(spec) == [
        Assumption(kind="place_scope", binding_ref="where:", text=text)
    ]


def test_a_role_the_scope_default_filled_is_assumed_whatever_its_member_count():
    """crs-002 names Bangladesh and no provider. With the provider scope defaulted to Japan
    alone, the heading reads "from Japan", and the role is still assumed."""
    spec = _panel_spec("crs-002")
    scope = ResolvedScope(
        axis="origin", members=[ResolvedEntity(ref="country/JPN", role="origin", label="Japan")]
    )
    scoped = spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"scope": scope})})
    assert _open_roles(assumptions_for(scoped)) == [_EVERY_PROVIDER]


def test_a_role_the_request_words_asked_every_member_of_is_no_assumption():
    """ "aid from all donors to Bangladesh": the words asked for every provider."""
    spec = _panel_spec("crs-002")
    assert _open_roles(assumptions_for(spec, asked_axes=frozenset({"origin"}))) == []


def test_a_candidates_part_assumes_what_every_reading_assumes():
    """ids-001's and ids-003's readings both read every creditor; beside a reading that
    names its creditor (ids-002), nothing is common."""
    specs = [_panel_spec("ids-001"), _panel_spec("ids-003")]
    lists = [assumptions_for(spec) for spec in specs]
    assert common_assumptions(lists) == [_EVERY_CREDITOR]
    assert common_assumptions([*lists, assumptions_for(_panel_spec("ids-002"))]) == []

    plan = candidates_sentence_plan(specs, assumptions=common_assumptions(lists))
    assert sentence_from_plan(plan).endswith(" Assumed: every creditor.")
    assert {t.segment for t in plan[-4:]} == {"assumption"}


def test_every_declared_default_the_index_holds_fits_an_echoed_assumption():
    """A resubmit echoes each part's assumptions as posted, so the bounds on `Assumption` hold
    every declared default the packaged index can fix: one value, named in at most the text
    bound by its label or short name."""
    from code_index.shared import get_index

    index = get_index()
    codes = [
        *(d.default.code for d in index.declarations.values() if d.default is not None),
        *(default.value_dcid for default in index.shape_slot_defaults),
    ]
    assert codes
    text_bound = Assumption.model_fields["text"].metadata
    max_chars = next(m.max_length for m in text_bound if getattr(m, "max_length", None))
    for code in codes:
        row = index.code_by_dcid(code)
        words = [code] if row is None else [row.label, row.short_name or row.label]
        assert all(len(word) <= max_chars for word in words), code


@pytest.mark.parametrize(
    "update",
    [
        pytest.param({"text": "x" * 201}, id="text"),
        pytest.param({"binding_ref": "how:" + "x" * 1033}, id="binding_ref"),
        pytest.param({"refs": ["CRSTypeOfFlow10", "CRSTypeOfFlow21"]}, id="refs"),
        pytest.param({"refs": ["x" * 1025]}, id="ref"),
    ],
)
def test_an_echoed_assumption_past_its_bounds_is_refused(update: dict[str, object]):
    """An echo is untrusted input, so each string and the value list is bounded."""
    item = {
        "kind": "declared_default",
        "binding_ref": "how:crsTypeOfFlow",
        "text": "ODA",
        "refs": ["CRSTypeOfFlow10"],
    }
    Assumption.model_validate(item)
    with pytest.raises(ValidationError):
        Assumption.model_validate({**item, **update})
