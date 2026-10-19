"""Pins that a release's bytes do not depend on the interpreter: the recorded slice, built and
staged in two processes with different `PYTHONHASHSEED` values, gives the same part hashes and
release id. A build in one process cannot catch output that follows set or dict iteration order
of strings, since both builds share one hash seed.

Each build runs in a fresh interpreter, so the test takes a few seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]

_BUILD = """
import json, sys, tempfile
from pathlib import Path

from graph_release.dfo import build_dfo_parts, stage_dfo_parts
from graph_release.format import compute_release_id
from graph_release.platform import stage_platform_parts
from tests.graph_release.recorded import (
    SLICE_BASE_PARTIES,
    SLICE_DECLARATIONS,
    slice_platform_parts,
)

with tempfile.TemporaryDirectory() as staging:
    parts = slice_platform_parts()
    staged = [
        *stage_platform_parts(parts, staging_dir=Path(staging), execution_id="exec-1"),
        *stage_dfo_parts(
            build_dfo_parts(parts, SLICE_DECLARATIONS, SLICE_BASE_PARTIES),
            staging_dir=Path(staging),
            execution_id="exec-1",
        ),
    ]
hashes = sorted([part.ref.name, part.ref.provenance or "", part.ref.sha256] for part in staged)
release_id = compute_release_id(
    part_hashes=(part.ref.sha256 for part in staged),
    import_execution_id="exec-1",
    dc_serving_revision="revision-1",
)
json.dump({"parts": hashes, "release_id": release_id}, sys.stdout)
"""


def _build(hash_seed: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _BUILD],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env=os.environ | {"PYTHONHASHSEED": hash_seed},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_release_bytes_do_not_depend_on_the_hash_seed():
    first, second = _build("1"), _build("2")

    assert {name for name, _, _ in first["parts"]} >= {"series", "edges", "code_index_blocks"}
    assert first["parts"] == second["parts"]
    assert first["release_id"] == second["release_id"]
