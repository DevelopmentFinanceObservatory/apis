"""Each candidate renders a distinct sentence that round-trips and marks its own variable."""

from __future__ import annotations

from search.models import CandidatesPart, RawTextInput, ResolveOptions, ResolveRequest
from tests.engine._harness import offline_resolve
from tests.render._helpers import assert_round_trip


def test_candidate_sentences_index_aligned_and_distinct():
    """The query returns distinct candidate sentences. Each plan round-trips, marks its `what`
    token as distinguishing, and leaves the place both readings share unmarked.
    """
    req = ResolveRequest(
        input=RawTextInput(query="remittances as percentage of GDP in Nigeria"),
        response_options=ResolveOptions(),
    )
    result = offline_resolve(req).parts[0]

    assert isinstance(result, CandidatesPart)
    specs = result.candidates.specs
    assert len(specs) >= 2

    rendered = []
    for spec in specs:
        assert spec.resolved.rendered_sentence is not None
        assert spec.resolved.sentence_plan is not None
        assert_round_trip(spec.resolved.sentence_plan, spec.resolved.rendered_sentence)
        rendered.append(spec.resolved.rendered_sentence)

        what_tok = next(t for t in spec.resolved.sentence_plan if t.axis == "what")
        assert what_tok.distinguishes is True
        for tok in spec.resolved.sentence_plan:
            if tok.axis == "where":
                assert tok.distinguishes is None
    assert len(set(rendered)) == len(rendered), (
        f"expected distinct candidate sentences, got {rendered}"
    )
