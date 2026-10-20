"""Checks what a resolved Spec assumed (`render.assumptions`) on Specs the sentence panel
recorded live, and the clause a candidates part's own sentence closes with.

The panel keeps no trace slot filters, so a test that needs a declared default records it
on the slot as the election does (`provenance="declared_default"`).
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


def test_aid_to_a_named_recipient_assumes_the_default_flow_and_every_provider():
    """crs-002, "aid to Bangladesh": the flow its declared default fixed, named as the code
    index names it since the measure noun says it, then the providers no word named."""
    spec = _with_declared_default(_panel_spec("crs-002"), "how:crsTypeOfFlow")
    assert assumptions_for(spec) == [
        Assumption(
            kind="declared_default",
            binding_ref="how:crsTypeOfFlow",
            text="ODA",
            refs=["CRSTypeOfFlow10"],
        ),
        _EVERY_PROVIDER,
    ]


def test_a_declared_default_the_heading_states_takes_the_heading_words():
    """crs-026: a purpose slot the heading states ("for malaria control") is named in the
    heading's words."""
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
        pytest.param("crs-004", [_EVERY_RECIPIENT, _EVERY_PROVIDER], id="crs-004 none named"),
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
    assert assumptions_for(_panel_spec(query_id)) == expected


def test_a_role_asked_member_by_member_is_no_assumption():
    """A role in open_roles, as "top 5 donors" opens it, was asked for."""
    spec = _panel_spec("crs-002")
    identity = spec.definition.identity.model_copy(update={"open_roles": ["origin"]})
    definition = spec.definition.model_copy(update={"identity": identity})
    assert assumptions_for(spec.model_copy(update={"definition": definition})) == []


@pytest.mark.parametrize(
    ("label", "members", "text"),
    [
        ("places", ["country/KEN", "geoId/06"], "every place"),
        ("countries", ["country/KEN", "country/UGA"], "every country"),
        ("worldwide", ["Earth"], "the world as a whole"),
    ],
)
def test_a_request_naming_no_place_assumes_the_place_scope_as_the_heading_counts_it(
    label: str, members: list[str], text: str
):
    """ "government debt in 2020" reads across 46 places: "every place", the heading's
    noun, and "every country" only where the heading counts countries."""
    spec = _spec(
        slots=[_what_slot("Gross public sector debt")],
        coverage=CoverageBreadth(
            has_data=True, window=None, dimensions=[BreadthDim(label=label, count=len(members))]
        ),
        scope=ResolvedScope(
            axis="observationAbout",
            members=[ResolvedEntity(ref=m, role="observationAbout") for m in members],
        ),
    )
    assert assumptions_for(spec) == [
        Assumption(kind="place_scope", binding_ref="where:", text=text)
    ]


def test_a_candidates_part_assumes_what_every_reading_assumes():
    """ids-003's two readings both read every creditor; beside a reading that names its
    creditor, nothing is common."""
    part = _panel_part("ids-003")
    assert isinstance(part, CandidatesPart)
    specs = part.candidates.specs
    assert common_assumptions(specs) == [_EVERY_CREDITOR]
    assert common_assumptions([*specs, _panel_spec("ids-002")]) == []

    plan = candidates_sentence_plan(specs, assumptions=common_assumptions(specs))
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
