"""Build the packaged code index with one edit applied to its document."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable

from code_index import default_index_path, parse_index
from code_index.export import content_hash
from code_index.query import CodeIndex


def packaged_index_with(edit: Callable[[dict], None]) -> CodeIndex:
    """Parse the packaged index document after `edit` changes it in place, restamped."""
    document = json.loads(gzip.decompress(default_index_path().read_bytes()))
    edit(document)
    blocks = {key: value for key, value in document.items() if key != "stamp"}
    document["stamp"]["content_hash"] = content_hash(blocks)
    return parse_index(document)
