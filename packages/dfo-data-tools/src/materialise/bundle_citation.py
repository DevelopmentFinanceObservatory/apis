"""Renders a bundle's CITATION.cff: a CFF 1.2.0 dataset citation as block YAML.

The DFO authors the extract. Each source is a `references` entry of `type: data`, authored by
its publisher, else by the DFO, since the schema requires authors. Every scalar is written as a
JSON string, which is a valid YAML 1.1 and 1.2 double-quoted scalar, so curated text cannot
change the document's structure. Characters YAML forbids unescaped are written as `\\u`
escapes. Values that would fail the CFF schema's URL or date patterns are left out.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC
from typing import TYPE_CHECKING

from materialise.bundle_naming import version_tag
from materialise.license_type import licence_ids

if TYPE_CHECKING:
    from materialise.bundle import BundleContext, SourceProvenance

_URL = re.compile(r"^(https|http|ftp|sftp)://\S+$")
"""The CFF 1.2.0 `url` pattern, with whitespace refused."""

_DATE = re.compile(r"^[0-9]{4}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])$")
"""The CFF 1.2.0 `date` pattern."""

_CFF_UNLISTED_SPDX = frozenset({"CC-BY-3.0-IGO"})
"""SPDX ids `materialise.license_type` maps that the CFF 1.2.0 licence enum (the SPDX list of
2021-05-14) lacks. Such a licence is cited by its URL."""

_YAML_UNPRINTABLE = re.compile("[\x7f-\x9f\ud800-\udfff﻿￾￿]")
"""Characters outside YAML's printable set, plus the byte-order mark, which `json.dumps` with
`ensure_ascii=False` leaves raw."""


def build_citation_cff(
    *,
    sources: Sequence[SourceProvenance],
    context: BundleContext,
    dfo_entity: Mapping[str, str],
) -> bytes:
    """The CFF 1.2.0 citation of the extract, with each of `sources` (the distinct sources
    across the bundle's members) as a reference. `dfo_entity` is the DFO's entity, its author
    and contact. The top-level `license` appears only when every source maps to the same SPDX
    id."""
    citation: dict[str, object] = {
        "cff-version": "1.2.0",
        "message": "If you use this dataset, please cite it as below.",
        "type": "dataset",
        "title": context.heading,
    }
    if context.summary:
        citation["abstract"] = context.summary
    citation |= {
        "version": version_tag(context.version),
        "date-released": context.prepared_at.astimezone(UTC).date().isoformat(),
    }
    if url := _url(context.links.site or context.links.version):
        citation["url"] = url
    spdx_ids = {_spdx(source) for source in sources}
    if len(spdx_ids) == 1 and (spdx := next(iter(spdx_ids))) is not None:
        citation["license"] = spdx
    citation |= {"authors": [dict(dfo_entity)], "contact": [dict(dfo_entity)]}
    if sources:
        citation["references"] = [_reference(source, dfo_entity=dfo_entity) for source in sources]
    return ("\n".join(_yaml_lines(citation, indent=0)) + "\n").encode("utf-8")


def _spdx(source: SourceProvenance) -> str | None:
    spdx = licence_ids(source.licence.type_id).spdx
    return None if spdx in _CFF_UNLISTED_SPDX else spdx


def _url(value: str | None) -> str | None:
    if value is None or not _URL.match(value.strip()):
        return None
    return value.strip()


def _reference(source: SourceProvenance, *, dfo_entity: Mapping[str, str]) -> dict[str, object]:
    documentation = source.documentation
    publisher = source.credited_publisher
    reference: dict[str, object] = {
        "type": "data",
        "title": source.name,
        "authors": [{"name": publisher}] if publisher is not None else [dict(dfo_entity)],
    }
    if url := _url(source.url):
        reference["url"] = url
    if spdx := _spdx(source):
        reference["license"] = spdx
    elif licence_url := _url(source.licence.url):
        reference["license-url"] = licence_url
    if documentation is None:
        return reference
    if documentation.last_update and _DATE.match(documentation.last_update):
        reference["date-accessed"] = documentation.last_update
    if documentation.attribution and documentation.attribution.strip():
        reference["notes"] = documentation.attribution.strip()
    return reference


def _scalar(value: str) -> str:
    quoted = json.dumps(value, ensure_ascii=False)
    return _YAML_UNPRINTABLE.sub(lambda match: f"\\u{ord(match.group()):04x}", quoted)


def _yaml_lines(value: object, *, indent: int) -> list[str]:
    """Block YAML for a mapping of strings, lists and mappings. A list item that is a mapping
    starts on the dash line, and its other keys align under the first."""
    pad = " " * indent
    lines: list[str] = []
    assert isinstance(value, Mapping)
    for key, item in value.items():
        if isinstance(item, str):
            lines.append(f"{pad}{key}: {_scalar(item)}")
            continue
        assert isinstance(item, list)
        lines.append(f"{pad}{key}:")
        for element in item:
            nested = _yaml_lines(element, indent=indent + 4)
            lines.append(f"{pad}  - {nested[0].lstrip()}")
            lines += nested[1:]
    return lines
