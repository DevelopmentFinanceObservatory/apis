"""Present a committed origin catalogue build as a catalogue root.

The builds under ``tests/fixtures`` are stored as a format root's contents, ``current.json``
and ``versions/``, and the loader reads ``<ORIGIN_CATALOGUE_URI>/<FORMAT_ROOT>/``. This links
a build into a root under the name ``origin_catalogue.schema.FORMAT_ROOT`` carries, and
prints the ``ORIGIN_CATALOGUE_URI`` that reads it.

Usage:
    uv run python scripts/stage_origin_catalogue_fixture.py <build-dir> <root-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from origin_catalogue.schema import FORMAT_ROOT


def stage_fixture(fixture_dir: Path, root: Path) -> str:
    """Links `fixture_dir` as `root`'s format root and returns `root`'s URI.
    Reads and writes under that URI land in `fixture_dir`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / FORMAT_ROOT).symlink_to(fixture_dir.resolve(strict=True), target_is_directory=True)
    return root.resolve().as_uri()


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(stage_fixture(Path(sys.argv[1]), Path(sys.argv[2])))


if __name__ == "__main__":
    main()
