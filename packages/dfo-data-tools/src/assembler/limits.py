"""Provisional resource maximums enforced by the assembler core."""

from __future__ import annotations

MAX_SERIES_PER_COMPONENT = 24
"""Resolved series one component may carry. `resolve.py` checks an estimate
of this against a selector's declared shape before it ever groups a frame's
rows into series, ahead of the cross-product expansion."""

MAX_POINTS_PER_SERIES = 500
"""Maximum periods per series at the frame's grain. Checked before binding.

The cap matches the catalogue's 500-point `line_chart` schema limit."""

MAX_TABLE_ROWS = 2_000
"""Maximum table rows, checked as `series_count * period_count` before binding.

The 2,000-row cap is below the theoretical 12,000 (24 series x 500 periods)
and matches the catalogue schema. Oversized tables are refused.
Accepted tables include every resolved row."""

MAX_SELECTORS = 16
"""Maximum selector count, checked before matching frame rows.

The write path currently normalizes `select` to one selector."""

LABEL_MAX = 200
"""Code points kept of every label a payload or its provenance carries, unit labels included."""

MAX_ENVELOPE_BYTES = 2 * 1024 * 1024
"""UTF-8 bytes allowed for `component` and `provenance`, using compact JSON with
`ensure_ascii=False`. Checked after minting and again on an in-transaction
rebuild, before a create pins its frame.

The four envelope scalars add under 200 bytes. Replies also carry a fact
summary capped at 2,000 approximate tokens. Readback trusts revisions stored
before this limit. A 2,000-row table with 200-character ASCII labels
uses about 1.24 MB. `LABEL_MAX` counts code points. Repeated multi-byte
labels can reach 9.6 MB. Unit labels are cut at `LABEL_MAX`. A source's `url` is
copied at full length."""


def joined_label(*parts: str, joiner: str) -> str:
    """Joins `parts` with `joiner` within `LABEL_MAX`, so every part and every joiner stays in
    the label.

    A join over `LABEL_MAX` shortens its longest parts to one shared length. A shorter part
    keeps its text, and the later of the shortened parts take any code points left over."""
    room = LABEL_MAX - len(joiner) * (len(parts) - 1)
    shortened = len(parts)
    for length in sorted(map(len, parts)):
        if length * shortened > room:
            break
        room -= length
        shortened -= 1
    else:
        return joiner.join(parts)
    share, spare = divmod(room, shortened)
    kept: list[str] = []
    for part in parts:
        if len(part) > share:
            shortened -= 1
            part = part[: share + (shortened < spare)]
        kept.append(part)
    return joiner.join(kept)


MAX_INTENT_BYTES = 65_536
"""Raw intent bytes, checked by the pipeline before schema validation."""

PER_DIMENSION_LISTING_CAP = 50
"""Distinct values `frame_view.compute_structural_view` lists per
dimension before truncating and reporting the full count in the tail's
place instead."""

FACT_SUMMARY_SERIES_CAP = 12
"""Series a `WriteReply.fact_summary` lists individually before folding the
remainder into a cut count."""

FACT_SUMMARY_TOKEN_CAP = 2_000
"""Approximate token budget for one `WriteReply.fact_summary`, estimated as
`len(text) // 4` -- coarse on purpose, to avoid a tokenizer dependency for a
provisional cap."""
