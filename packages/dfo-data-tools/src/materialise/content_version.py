"""Builds materialisation content-version preimages and delegates minting to
`dfo_frame_grammar.content_version`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from dfo_frame_grammar import (
    FRAME_GRAMMAR_VERSION,
    Frame,
    FrameTrace,
    mint_content_version,
    recompute_content_version,
    stream_frame_sha256,
)

from materialise.convert import ConversionRecord
from materialise.recipe_bodies import RECIPE_BODIES, recipe_body_digest

# Disjoint source-key prefix for a converted member's native-unit revision.
_NATIVE_UNIT_SOURCE_PREFIX = "native_unit/"

# Disjoint source-key prefix for recipe-body revisions.
_RECIPE_BODY_SOURCE_PREFIX = "recipe_body/"


def facet_preimage(*, axis: str, ref: str, basis: str) -> dict[str, object]:
    """`{"axis": axis, "ref": ref, "basis": basis}`. `ref` is always a plain
    string.
    """
    return {"axis": axis, "ref": ref, "basis": basis}


def source_revision_preimage(*, source: str, revision: str) -> dict[str, object]:
    """`{"source": source, "revision": revision}`."""
    return {"source": source, "revision": revision}


def native_unit_source_revision(*, stat_var: str, native_unit: str) -> dict[str, object]:
    """Preimage for a converted member's native unit: `native_unit/<stat_var>` maps to
    the native unit dcid (for example, `CurrencyUSD`). `stat_var` is bare, without `dcid:`.
    This revision makes a changed native-unit graph declaration change `content_version`,
    even when the converted rows and unit request stay unchanged."""
    return source_revision_preimage(
        source=f"{_NATIVE_UNIT_SOURCE_PREFIX}{stat_var}", revision=native_unit
    )


def recipe_body_source_revision(conversion: ConversionPreimage) -> dict[str, object] | None:
    """Revision for `RECIPE_BODIES[recipe_id]`, keyed as `recipe_body/<recipe_id>` and
    hashed from the recipe body. Returns `None` when no body exists. Production recipe
    resolution refuses that id. A body change under the same id changes `content_version`."""
    recipe = RECIPE_BODIES.get(conversion.recipe_id)
    if recipe is None:
        return None
    return source_revision_preimage(
        source=f"{_RECIPE_BODY_SOURCE_PREFIX}{conversion.recipe_id}",
        revision=recipe_body_digest(recipe),
    )


def _with_recipe_body_revision(
    source_revisions: Iterable[Mapping[str, object]], conversion: ConversionPreimage
) -> tuple[Mapping[str, object], ...]:
    """Returns `source_revisions` with the `recipe_body/<recipe_id>` revision appended when no
    entry already names that source. Finish writes the revision into a converted frame's trace,
    so a client recomputing through the grammar alone reproduces the token. Traces stored before
    that lack it, and the grammar refuses a duplicate `source`, so the append is conditional.
    A recipe id with no body adds nothing."""
    revisions = tuple(source_revisions)
    entry = recipe_body_source_revision(conversion)
    if entry is None or any(existing["source"] == entry["source"] for existing in revisions):
        return revisions
    return (*revisions, entry)


def recipe_dependency_fingerprint(recipe_id: str) -> tuple[str, str] | None:
    """Returns `(recipe_id, recipe_body_digest)` for the current body, or `None` when
    `RECIPE_BODIES` has no entry. Materialisation and currency rechecks use this same rule."""
    recipe = RECIPE_BODIES.get(recipe_id)
    if recipe is None:
        return None
    return recipe_id, recipe_body_digest(recipe)


@dataclass(frozen=True)
class ConversionPreimage:
    """The three `ConversionRecord` fields used for identity: `recipe_id`,
    `exchange_rate_table_version`, and `deflator_table_version`. Other record fields describe
    execution metadata. Minting reads only these identity fields. `from_conversion_record`
    projects this shape from a live record."""

    recipe_id: str
    exchange_rate_table_version: str
    deflator_table_version: str | None

    @classmethod
    def from_conversion_record(cls, record: ConversionRecord | None) -> ConversionPreimage | None:
        """Projects a record to the three identity fields. `None` returns `None`."""
        if record is None:
            return None
        return cls(
            recipe_id=record.recipe_id,
            exchange_rate_table_version=record.exchange_rate_table_version,
            deflator_table_version=record.deflator_table_version,
        )


def _conversion_preimage(conversion: ConversionPreimage) -> dict[str, object]:
    """Maps to the grammar's `{recipe, rate_table_version, deflator_table_version}` shape.
    Omits `deflator_table_version` when `None` because current-price conversion joins no
    deflator. An empty string counts as an explicit version. This is materialise's only map
    using the grammar's key names."""
    preimage: dict[str, object] = {
        "recipe": conversion.recipe_id,
        "rate_table_version": conversion.exchange_rate_table_version,
    }
    if conversion.deflator_table_version is not None:
        preimage["deflator_table_version"] = conversion.deflator_table_version
    return preimage


@dataclass(frozen=True)
class MintPreimage:
    """`mint_version` inputs except `frame`, captured so a transformed frame can reuse
    the same identity preimage. `source_revisions` includes graph, native-unit, and recipe-body
    revisions. A preimage rebuilt from a trace stored without the recipe-body revision gets it
    from `mint_version` (`_with_recipe_body_revision`). `conversion=None` means no conversion."""

    graph_revision: str
    source_revisions: tuple[dict[str, object], ...]
    facets: tuple[dict[str, object], ...]
    conversion: ConversionPreimage | None = None


def mint_preimage_from_trace(trace: Mapping[str, object]) -> MintPreimage | None:
    """Rebuilds mint inputs solely from stored trace data. Returns `None` when `mint_facets`
    is absent. Wider trace facets store resolved native
    units, while `mint_facets` preserves the converted unit's original request ref. Callers
    rematerialise when that field is absent.

    `graph_revision` and `source_revisions` come from the trace. `mint_facets` preserves the
    exact `{axis, ref, basis}` preimage used at finish. Conversion carries recipe, rate-table,
    and optional deflator versions from the trace's one `conversion` entry. A converted trace
    carries its `recipe_body/<recipe>` revision; for a trace stored without it, `mint_version`
    adds it (`_with_recipe_body_revision`).

    Raises:
        KeyError: The trace lacks `graph_revision` or `source_revisions`, or its conversion
            entry lacks `recipe` or `rate_table_version`.
        ValueError: `conversion` is not a one-entry list. A definition converts under one
            recipe and table-version pair, so any other shape is corrupt stored data.
            Migrations `0017_trace_conversion_entries.sql` and
            `0020_trace_conversion_entries_after_deploy.sql` rewrote every stored object
            shape that predates the list."""
    mint_facets_entries = trace.get("mint_facets")
    if mint_facets_entries is None:
        return None
    facets = tuple(
        dict(entry) for entry in cast("Iterable[Mapping[str, object]]", mint_facets_entries)
    )
    conversion_entries = trace.get("conversion")
    conversion: ConversionPreimage | None = None
    if conversion_entries is not None:
        if not isinstance(conversion_entries, list) or len(conversion_entries) != 1:
            raise ValueError(
                f"stored trace conversion is not a one-entry list: {conversion_entries!r}"
            )
        conversion_entry = cast("Mapping[str, object]", conversion_entries[0])
        conversion = ConversionPreimage(
            recipe_id=cast(str, conversion_entry["recipe"]),
            exchange_rate_table_version=cast(str, conversion_entry["rate_table_version"]),
            deflator_table_version=cast(
                "str | None", conversion_entry.get("deflator_table_version")
            ),
        )
    return MintPreimage(
        graph_revision=cast(str, trace["graph_revision"]),
        source_revisions=tuple(
            dict(entry)
            for entry in cast("Iterable[Mapping[str, object]]", trace["source_revisions"])
        ),
        facets=facets,
        conversion=conversion,
    )


def mint_version(
    frame: Frame,
    *,
    graph_revision: str,
    source_revisions: Iterable[Mapping[str, object]],
    facets: Iterable[Mapping[str, object]],
    conversion: ConversionPreimage | None = None,
) -> str:
    """Single-frame convenience wrapper around `mint_version_from_chunks`."""
    return mint_version_from_chunks(
        (frame,),
        graph_revision=graph_revision,
        source_revisions=source_revisions,
        facets=facets,
        conversion=conversion,
    )


def mint_version_from_chunks(
    chunks: Iterable[Frame],
    *,
    graph_revision: str,
    source_revisions: Iterable[Mapping[str, object]],
    facets: Iterable[Mapping[str, object]],
    conversion: ConversionPreimage | None = None,
) -> str:
    """Hashes validated grammar `Frame` chunks in iteration order, then mints their version.
    Accepts grammar `Frame` chunks built by `Frame.build`, which normalises each cell once. A
    single chunk hashes like `frame_sha256`. Multiple chunks hash like their concatenated rows
    while holding at most one chunk's rows in memory.

    `conversion=None` means no recipe and is passed to the grammar as an omitted argument.
    Explicit `conversion=None` there raises. A supplied `ConversionPreimage` is mapped to the
    grammar's keys here, and its recipe-body revision is added when available and absent
    from `source_revisions` (`_with_recipe_body_revision`). Use
    `mint_version_from_digest` when the frame digest is already computed."""
    digest = stream_frame_sha256(chunks)
    return mint_version_from_digest(
        digest,
        graph_revision=graph_revision,
        source_revisions=source_revisions,
        facets=facets,
        conversion=conversion,
    )


def mint_version_from_digest(
    digest: str,
    *,
    graph_revision: str,
    source_revisions: Iterable[Mapping[str, object]],
    facets: Iterable[Mapping[str, object]],
    conversion: ConversionPreimage | None = None,
) -> str:
    """Mints from an existing frame digest and the same revision preimage as
    `mint_version_from_chunks`. Group generation hashes during its single fetch, before
    the revision snapshot is known. `_finish_group` uses this function to avoid rereading rows."""
    if conversion is None:
        return mint_content_version(
            frame_sha256=digest,
            graph_revision=graph_revision,
            source_revisions=source_revisions,
            facets=facets,
        )
    return mint_content_version(
        frame_sha256=digest,
        graph_revision=graph_revision,
        source_revisions=_with_recipe_body_revision(source_revisions, conversion),
        facets=facets,
        conversion=_conversion_preimage(conversion),
    )


def frame_trace_fragment(
    *,
    graph_revision: str,
    source_revisions: Iterable[Mapping[str, object]],
    facets: Iterable[Mapping[str, object]],
    frame_grammar_version: str = FRAME_GRAMMAR_VERSION,
) -> FrameTrace:
    """Builds a `FrameTrace` recompute witness with `conversion=None` (no recipe)."""
    return FrameTrace(
        graph_revision=graph_revision,
        source_revisions=tuple(source_revisions),
        facets=tuple(facets),
        frame_grammar_version=frame_grammar_version,
        conversion=None,
    )


def recompute_version(canonical_bytes: bytes, fragment: FrameTrace) -> str:
    """Recomputes the content version from canonical frame bytes and a `FrameTrace`."""
    return recompute_content_version(canonical_bytes, fragment)
