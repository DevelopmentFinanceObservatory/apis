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

_ORG_PACK_ID = "org"
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


def _is_listed_name(normalized_text: str, entity: EntityRecord) -> bool:
    if normalized_text == entity.canonical_name_norm:
        return True
    if any(normalized_text == name.value_norm for name in entity.names):
        return True
    return any(normalized_text == code.value_norm for code in entity.codes)


def filter_listed_candidates(
    resolver: Resolver, text: str, candidates: Sequence[CandidateSummary]
) -> list[CandidateSummary]:
    """Return named candidates backed by resolver entities, ordered by confidence then ID."""
    survivors = [
        candidate
        for candidate in candidates
        if (entity := resolver.entity(candidate.entity_id)) is not None
        and _is_listed_name(_pack_normalize(resolver, text, candidate), entity)
    ]
    return sorted(
        survivors,
        key=lambda c: (-(c.confidence if c.confidence is not None else 0.0), c.entity_id),
    )
