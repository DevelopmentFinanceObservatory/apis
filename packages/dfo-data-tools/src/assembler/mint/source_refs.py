"""How a component names its sources: each `source_ref` and `source_refs` entry.

A card with a description names a source by its dcid, the `ref` of an entry in
`description.sources`. A card without one names it by the `src_` id of an entry in
`provenance.sources`, which records the dcid as its `ref`. `check_source_refs` refuses an
envelope where an entry names neither, and `renamed_source_refs` over `source_dcids` turns the
second form into the first.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import cast

from assembler.errors import SchemaInvalid

_SOURCE_REF = "source_ref"
_SOURCE_REFS = "source_refs"


def source_refs(value: object, pointer: str = "") -> Iterator[tuple[str, str]]:
    """Each `source_ref` and `source_refs` entry at any depth of `value`, with its JSON
    pointer under `pointer`."""
    if isinstance(value, Mapping):
        for key, item in cast("Mapping[str, object]", value).items():
            at = f"{pointer}/{key}"
            if key == _SOURCE_REF:
                yield at, cast(str, item)
            elif key == _SOURCE_REFS:
                for n, ref in enumerate(cast("Sequence[str]", item)):
                    yield f"{at}/{n}", ref
            else:
                yield from source_refs(item, at)
    elif isinstance(value, list):
        for n, item in enumerate(cast("list[object]", value)):
            yield from source_refs(item, f"{pointer}/{n}")


def renamed_source_refs(value: object, names: Mapping[str, str]) -> object:
    """`value` with each `source_ref` and `source_refs` entry that `names` holds replaced by
    its name there."""
    if isinstance(value, Mapping):
        renamed: dict[str, object] = {}
        for key, item in cast("Mapping[str, object]", value).items():
            if key == _SOURCE_REF:
                renamed[key] = names.get(cast(str, item), item)
            elif key == _SOURCE_REFS:
                renamed[key] = [names.get(ref, ref) for ref in cast("Sequence[str]", item)]
            else:
                renamed[key] = renamed_source_refs(item, names)
        return renamed
    if isinstance(value, list):
        return [renamed_source_refs(item, names) for item in cast("list[object]", value)]
    return value


def source_dcids(provenance: Mapping[str, object]) -> dict[str, str]:
    """Each `provenance.sources[]` id that records its source's dcid, mapped to it."""
    sources = cast("Sequence[Mapping[str, object]]", provenance.get("sources") or ())
    return {
        cast(str, source["id"]): cast(str, source["ref"]) for source in sources if "ref" in source
    }


def check_source_refs(envelope: Mapping[str, object]) -> None:
    """Refuses `envelope` when a component entry names no source the envelope lists: on a card
    with a description, a `description.sources[].ref`; otherwise a `provenance.sources[].id`.

    Raises:
        SchemaInvalid: at the first entry that names no listed source.
    """
    description = cast("Mapping[str, object] | None", envelope.get("description"))
    if description is not None:
        listed = cast("Sequence[Mapping[str, object]]", description["sources"])
        named, where = {source.get("ref") for source in listed}, "description.sources[].ref"
    else:
        provenance = cast("Mapping[str, object]", envelope["provenance"])
        listed = cast("Sequence[Mapping[str, object]]", provenance.get("sources") or ())
        named, where = {source["id"] for source in listed}, "provenance.sources[].id"
    for pointer, ref in source_refs(envelope["component"], "/component"):
        if ref not in named:
            raise SchemaInvalid(
                pointer=pointer,
                detail=f"resolved payload names source {ref!r}, which no {where} entry names",
                legal_fields=(),
            )
