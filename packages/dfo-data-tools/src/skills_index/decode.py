"""Decodes the graph's JSON-escaped `text` arc into Markdown, once, at the read.

The publishing pipeline stores every AgentSkill `text` value (a skill body and
a reference's content) as the inside of a JSON string. A newline is the two
characters `\\n` and a quote is the six characters `\\u0022`.
`DecodedTextGraph` decodes that arc as `node_arcs_many` returns it, so the
build, its byte budgets, and everything served afterwards see Markdown. It
refuses a stored value longer than `JSON_ESCAPE_EXPANSION_FACTOR` times its
decoded byte limit before decoding, since such a value cannot decode within
the limit. Other arcs (`name`, `description`, `url`) are stored as written
and pass through unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

from dfo_provenance_docs import DocumentError, decode_mcf_text

from skills_index.config import (
    CONTENT_ARC,
    JSON_ESCAPE_EXPANSION_FACTOR,
    OVERSIZED_CONTENT_ARC,
    UNDECODABLE_CONTENT_ARC,
)
from skills_index.graph import SkillsGraph, SubjectsPage


def _decode_content_arc(
    node_arcs: dict[str, tuple[str, ...]], *, max_decoded_bytes: int
) -> dict[str, tuple[str, ...]]:
    """Returns `node_arcs` with its `text` values decoded.

    A stored value longer than the escaped form of `max_decoded_bytes` leaves
    `text` empty and records why under `OVERSIZED_CONTENT_ARC`, without
    decoding any value. A value that does not decode leaves `text` empty and
    records the decoder's explanation under `UNDECODABLE_CONTENT_ARC`, so the
    build tells an oversized or undecodable node from one that carries no text.
    """
    values = node_arcs.get(CONTENT_ARC, ())
    max_stored_chars = JSON_ESCAPE_EXPANSION_FACTOR * max_decoded_bytes
    if (longest := max(map(len, values), default=0)) > max_stored_chars:
        detail = (
            f"is {longest} stored characters, past the {max_stored_chars}-character "
            f"ceiling of an escaped {max_decoded_bytes}-byte text"
        )
        return {**node_arcs, CONTENT_ARC: (), OVERSIZED_CONTENT_ARC: (detail,)}
    try:
        decoded = tuple(decode_mcf_text(value) for value in values)
    except DocumentError as exc:
        return {**node_arcs, CONTENT_ARC: (), UNDECODABLE_CONTENT_ARC: (exc.detail,)}
    return {**node_arcs, CONTENT_ARC: decoded}


class DecodedTextGraph:
    """A `SkillsGraph` that decodes the `text` arc of every `node_arcs_many` read.

    `max_decoded_bytes` is the decoded limit of the text this graph reads,
    `SKILL_BODY_MAX_BYTES` for bodies or `SKILL_REFERENCE_MAX_BYTES` for
    references. It sets the stored-length ceiling checked before decoding.
    """

    def __init__(self, graph: SkillsGraph, *, max_decoded_bytes: int) -> None:
        self._graph = graph
        self._max_decoded_bytes = max_decoded_bytes

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        arcs = self._graph.node_arcs_many(dcids, properties=properties, deadline=deadline)
        if CONTENT_ARC not in properties:
            return arcs
        return {
            dcid: None
            if node_arcs is None
            else _decode_content_arc(node_arcs, max_decoded_bytes=self._max_decoded_bytes)
            for dcid, node_arcs in arcs.items()
        }

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        return self._graph.node_subjects_many(dcids, property=property, max_results=max_results)
