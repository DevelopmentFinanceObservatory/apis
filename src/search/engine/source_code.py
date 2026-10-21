"""Resolve a part's publisher series code (`ParsedPart.source_code`) to the StatVars it names.

The release's code index records each series code the StatVars it lists carry on their
`sourceIndicatorCode` arcs, with the StatVars that carry it under each provenance holding it
(`CodeIndex.source_code_members`). A code's members replace recall's embedding resolve
(`interpret.recall`), and shape derivation, the twin election and bind run on them as on a
recalled pool. "debt service" under DT.TDS.DEGG.CD elects the by-creditor variable when the
part opens the creditor role and the aggregate when it does not.

A code compiles at the door, before any read, against the provenances the part's sources
allow (`compile_part_code`). It refuses `source_code_not_held` when no allowed provenance
holds it, `source_code_ambiguous` when two allowed provenances hold it and the part names no
source, and `source_code_shares_variable` when its source gives one of its StatVars another
code too, so the StatVar cannot stand for the code alone (IDS gives one net-flow StatVar both
DT.NFL.IMFC.CD and DT.NFL.MIDA.CD). A compiled code whose StatVars cannot answer the part
refuses `source_code_not_held` later, from the part's pipeline (`ResolveState.no_data`).
Each `source_code_not_held` names its cause (`NotHeldCause`).

On the raw-text door the extraction names no code, and `scan_source_codes` finds one in the
query's own words.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from code_index.query import CodeIndex
from search.engine.retrieval.bound_values import elects_within
from search.engine.source_scope import SourceReference, SourceScope, source_label

if TYPE_CHECKING:
    from search.models import BaseIdentity

type SourceCodeRefusalReason = Literal[
    "source_code_not_held", "source_code_ambiguous", "source_code_shares_variable"
]

type NotHeldCause = Literal["unknown", "excluded", "not_served", "unanswerable"]
"""Why a part's series code refuses `source_code_not_held`: the index holds no such code
(`unknown`); the part's sources leave out every provenance holding it (`excluded`); the
request's catalogue admits, or the graph confirms, none of its StatVars (`not_served`); or
its StatVars cannot answer the part as asked (`unanswerable`)."""

# A token runs between whitespace, brackets, quotes and the separators prose puts between
# words. `.`, `_` and `-` stay inside it, as series codes write them.
_TOKEN = re.compile("[^\\s()\\[\\]{}<>\"'\u2018\u2019\u201c\u201d,;:!?]+")
# The characters that make a token read as a code whatever its case.
_CODE_MARKS = re.compile(r"[._\d]")


@dataclass(frozen=True, slots=True)
class SeriesCodeMembers:
    """The StatVars `dcids`, sorted, that the provenance `provenance` gives the series code
    `code`, upper-cased."""

    code: str
    provenance: str
    dcids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceCodeRefusal:
    """A part whose series code `code` resolves to no StatVar it can answer from, so it
    answers no_data with `reason`.

    `cause` says why a `source_code_not_held` refusal holds, and is `None` for the others.
    `sources` are the provenances the refusal names: for `source_code_not_held`, every one
    whose codes the index holds when the cause is `unknown` and the ones holding the code
    otherwise; for `source_code_ambiguous`, the part's allowed ones holding the code.
    `shared_codes` are the other codes the code's source gives its StatVars, for
    `source_code_shares_variable`."""

    reason: SourceCodeRefusalReason
    code: str
    cause: NotHeldCause | None = None
    sources: tuple[SourceReference, ...] = ()
    shared_codes: tuple[str, ...] = ()


type PartCode = SeriesCodeMembers | SourceCodeRefusal
"""What a part's series code compiles to (`compile_part_code`)."""


def _provenance_reference(provenance: str) -> SourceReference:
    return SourceReference(ref=provenance, label=source_label(provenance), kind="provenance")


def not_held(code: str, *, cause: NotHeldCause, index: CodeIndex) -> SourceCodeRefusal:
    """The `source_code_not_held` refusal of `code` for `cause`, naming the provenances
    `SourceCodeRefusal.sources` describes."""
    provenances = (
        sorted(index.source_code_provenances)
        if cause == "unknown"
        else list(index.source_code_members(code))
    )
    return SourceCodeRefusal(
        reason="source_code_not_held",
        code=code,
        cause=cause,
        sources=tuple(_provenance_reference(p) for p in provenances),
    )


def compile_part_code(code: str, *, scope: SourceScope | None, index: CodeIndex) -> PartCode:
    """The StatVars the series code `code` names under the one provenance holding it that
    `scope` allows (every provenance when `scope` is `None`), or the refusal the module
    docstring describes."""
    holding = {
        provenance: dcids
        for provenance, dcids in index.source_code_members(code).items()
        if scope is None or provenance in scope.allowed
    }
    if not holding:
        cause = "excluded" if index.source_code_members(code) else "unknown"
        return not_held(code, cause=cause, index=index)
    if len(holding) > 1:
        return SourceCodeRefusal(
            reason="source_code_ambiguous",
            code=code,
            sources=tuple(_provenance_reference(p) for p in holding),
        )
    ((provenance, dcids),) = holding.items()
    if shared := index.shared_codes(provenance, code):
        return SourceCodeRefusal(
            reason="source_code_shares_variable", code=code, shared_codes=shared
        )
    return SeriesCodeMembers(code=code, provenance=provenance, dcids=dcids)


def stands_for_code(
    code: str,
    *,
    provenance: str,
    identity: BaseIdentity,
    served: Collection[str],
    read: str | None,
    index: CodeIndex,
) -> bool:
    """Whether a part naming the series code `code` and `provenance` as its source accepts
    as its answer the Spec of `identity`, which serves the StatVars `served` from the
    provenance `read`, so the Spec is the series `provenance` files under `code` and no
    other.

    The Spec reads `provenance` alone, where a part naming it reads only it. `provenance`
    gives none of the code's StatVars another code, as `compile_part_code` requires. Each of
    `served` is one of the code's, as recall from the code's StatVars leaves them. The
    StatVars the Datasets mint elects for `identity` are known and all the code's
    (`bound_values.elects_within`), as the part's pipeline requires
    (`region.pipeline._within_source_code`). The election runs last, and for one code of a
    Spec at most, since two unshared codes hold no StatVar in common."""
    members = index.source_code_members(code).get(provenance, ())
    return (
        read == provenance
        and not index.shared_codes(provenance, code)
        and set(served) <= set(members)
        and elects_within(identity, members)
    )


def _less_code_words(variable: str, code: str) -> str:
    """`variable` less each token that writes `code`, in any case."""
    pattern = r"(?<![\w.])" + re.escape(code) + r"(?![\w])"
    return " ".join(re.sub(pattern, " ", variable, flags=re.IGNORECASE).split())


def without_code_words(variable: str, code: str) -> str:
    """`variable` less each token that writes `code`, in any case. "debt service
    DT.TDS.DEGG.CD" reads "debt service". A variable made only of the code stays as it is."""
    return _less_code_words(variable, code) or variable


def names_only_code(variable: str, code: str) -> bool:
    """Whether `variable` holds no word but those writing `code`, or none at all."""
    return not _less_code_words(variable, code)


def _bare(token: str) -> str:
    """`token` without the full stops prose ends a sentence with."""
    return token.rstrip(".")


def _reads_as_code(token: str, held: str) -> bool:
    """Whether `token` writes the held code `held`: in any case when it holds `.`, `_` or a
    digit, else only in capitals, so "lp" and "Lp" in running text name no code LP."""
    return token == held or (token.upper() == held and _CODE_MARKS.search(token) is not None)


def scan_source_codes(query: str, *, measures: Sequence[str], index: CodeIndex) -> list[str | None]:
    """The series code `query` writes for each of `measures`, parallel to them, `None` for a
    measure it writes none for.

    A token of the query names a code when it writes a code `index` holds (`_reads_as_code`).
    The code qualifies the measure whose words hold the token, else the only measure, else
    no measure. A measure keeps the first code the query writes for it."""
    codes: list[str | None] = [None] * len(measures)
    measure_tokens = [{_bare(t).upper() for t in _TOKEN.findall(m)} for m in measures]
    for match in _TOKEN.finditer(query):
        token = _bare(match.group())
        held = token.upper()
        if not token or not index.source_code_members(held) or not _reads_as_code(token, held):
            continue
        targets = [i for i, tokens in enumerate(measure_tokens) if held in tokens]
        if not targets and len(measures) == 1:
            targets = [0]
        for position in targets:
            if codes[position] is None:
                codes[position] = held
    return codes
