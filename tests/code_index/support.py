"""The recorded code index document the session release and the index tests read, the
readers that parse an index document from a file, and the graph check of the implied values
no index row backs."""

import gzip
import json
from collections.abc import Sequence
from pathlib import Path

from code_index.export import ExportReader
from code_index.loader import parse_index, parse_measure_axes
from code_index.query import CodeIndex
from code_index.types import ImpliedValue, MeasureIndex

INDEX_FIXTURE = Path(__file__).parents[1] / "fixtures" / "code_index" / "index.json.gz"
"""A gzip code index export of the dev graph; `load_index` and `load_measure_axes` read it."""

_GZIP_MAGIC = b"\x1f\x8b"


def load_index(source: Path | str) -> CodeIndex:
    """`parse_index` of the plain or gzip JSON document at `source`."""
    return parse_index(_read_document(source), source=source)


def load_measure_axes(source: Path | str) -> MeasureIndex:
    """`parse_measure_axes` of the plain or gzip JSON document at `source`."""
    return parse_measure_axes(_read_document(source), source=source)


def _read_document(source: Path | str) -> dict:
    """The JSON document at `source`, decompressed first when it carries the gzip magic
    number."""
    raw = Path(source).read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    return json.loads(raw)


def unstated_implied_values(
    reader: ExportReader, implied_values: Sequence[ImpliedValue]
) -> list[ImpliedValue]:
    """The declarations of `implied_values` no StatVar of their measure states in `reader`'s
    graph: no StatVar of the declared `populationType` and `measuredProperty` carries the value
    on the property, so the value is no reading of the measure. The index holds no row of a
    catalogue family such as life expectancy, so the loader cannot check these
    (`code_index.loader._check_implied_values`)."""
    return [
        declared
        for declared in implied_values
        if not reader.node_subjects(
            declared.value_dcid,
            property=declared.property_dcid,
            filters={
                "populationType": declared.population_type,
                "measuredProperty": declared.measured_property,
            },
            max_results=1,
        ).dcids
    ]
