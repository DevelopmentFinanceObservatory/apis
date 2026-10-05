"""Write the sample origin catalogue build to ``tests/fixtures/origin_catalogue``.

Search loads an origin catalogue at startup, so the docker job in ``ci.yml`` mounts this
build into its read-only container. The build holds
``tests.origin_catalogue.support.sample_tables()``, encoded and published by the catalogue
job's own writer (``origin_catalogue.artifact``), and
``tests/origin_catalogue/test_fixture.py`` fails when the committed files stop matching it.
The directory holds a format root's contents, which
``scripts/stage_origin_catalogue_fixture.py`` presents as a catalogue root.

Usage:
    uv run python scripts/record_origin_catalogue_fixture.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage_origin_catalogue_fixture import stage_fixture  # noqa: E402
from tests.origin_catalogue.support import (  # noqa: E402
    FIXTURE_BUILD_ID,
    FIXTURE_DIR,
    publish_sample,
)


def main() -> None:
    # `publish_build` writes each build file once, so the old build goes first.
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    FIXTURE_DIR.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as root:
        publish_sample(stage_fixture(FIXTURE_DIR, Path(root)), build_id=FIXTURE_BUILD_ID)
    print(f"wrote build {FIXTURE_BUILD_ID!r} to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
