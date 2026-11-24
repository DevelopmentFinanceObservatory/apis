"""Keep only resolvekit candidates named by the query, then rank the survivors.

Each candidate uses its pack's normalizer. This prevents resolvekit's cross-pack preference for
`RESOLVED` over `AMBIGUOUS` from discarding stronger listed matches. For “World Bank”, IBRD and
IDA survive even when the geo pack's income-group result wins the merge.
"""

from __future__ import annotations

from collections.abc import Sequence

from resolvekit import EntityRecord, Resolver
from resolvekit.core.model.result import CandidateSummary
from resolvekit.packs.org.normalizer import OrgNormalizer

from entity_resolution.types import MatchedAs

_ORG_PACK_ID = "org"
# The name kind resolvekit gives a record's canonical name in each language.
_CANONICAL = "canonical"
_org_normalizer = OrgNormalizer()


class GateError(Exception):
    """Resolvekit contract violation from a candidate with no configured pack normaliser."""


def _pack_normalize(resolver: Resolver, text: str, candidate: CandidateSummary) -> str:
    """Normalize `text` to compare with the candidate's stored names and codes.

    Org records use `OrgNormalizer.normalize_name` because it matches their stored `value_norm`
    fields, including punctuation in aliases such as `Africa, regional`."""
    if candidate.pack_id == _ORG_PACK_ID:
        return _org_normalizer.normalize_name(text)
    normalizer = resolver.pack_normalizers.get(candidate.pack_id)
    if normalizer is None:
        raise GateError(
            f"candidate {candidate.entity_id!r} names pack {candidate.pack_id!r}, which this "
            "resolver has no normaliser for"
        )
    return normalizer.normalize(text)


def _listed_as(normalized_text: str, entity: EntityRecord) -> MatchedAs | None:
    """How `entity` lists the text: its canonical name, then a code, then any other name.

    A text a record holds both as a code and as an alias (`KE` for Kenya) matches as `code`."""
    if normalized_text == entity.canonical_name_norm or any(
        normalized_text == name.value_norm and name.kind == _CANONICAL for name in entity.names
    ):
        return "name"
    if any(normalized_text == code.value_norm for code in entity.codes):
        return "code"
    if any(normalized_text == name.value_norm for name in entity.names):
        return "alias"
    return None


def filter_listed_candidates(
    resolver: Resolver, text: str, candidates: Sequence[CandidateSummary]
) -> list[tuple[CandidateSummary, MatchedAs]]:
    """Return named candidates backed by resolver entities, each with how its record lists
    `text`, ordered by confidence then ID."""
    survivors = [
        (candidate, matched_as)
        for candidate in candidates
        if (entity := resolver.entity(candidate.entity_id)) is not None
        and (matched_as := _listed_as(_pack_normalize(resolver, text, candidate), entity))
        is not None
    ]
    return sorted(
        survivors,
        key=lambda pair: (
            -(pair[0].confidence if pair[0].confidence is not None else 0.0),
            pair[0].entity_id,
        ),
    )
