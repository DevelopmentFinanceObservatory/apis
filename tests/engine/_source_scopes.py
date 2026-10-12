"""Source scopes a test builds for a part without compiling its mentions."""

from __future__ import annotations

from collections.abc import Collection

from code_index.shared import get_index
from search.engine.source_scope import SourceReference, SourceScope, source_label
from search.models import SourceMention


def part_scope(named: str | None = None, excluded: Collection[str] = ()) -> SourceScope | None:
    """The scope of a part that names the declared dataset `named` by its ref, or excludes
    the declared datasets `excluded`, over the code index's declared datasets; `None` for a
    part that does neither."""
    if named is None and not excluded:
        return None
    universe = frozenset(get_index().datasets) if named is None else frozenset({named})
    mentions = [
        *([SourceMention(ref=named)] if named is not None else []),
        *(SourceMention(ref=dataset, polarity="exclude") for dataset in sorted(excluded)),
    ]
    return SourceScope(
        allowed=universe - frozenset(excluded),
        named=named is not None,
        mentions=tuple(mentions),
        references=tuple(
            SourceReference(ref=ref, label=source_label(ref), kind="provenance")
            for ref in (mention.ref for mention in mentions)
            if ref is not None
        ),
        expansions=tuple(frozenset({mention.ref}) for mention in mentions if mention.ref),
    )
