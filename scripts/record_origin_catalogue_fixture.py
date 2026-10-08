"""Write the committed origin catalogue builds under ``tests/fixtures``.

Search loads an origin catalogue at startup, so the docker job in ``ci.yml`` mounts the sample
build, ``tests/fixtures/origin_catalogue``, into its read-only container. The sample holds
``tests.origin_catalogue.support.sample_tables()`` and the UN tables of
``sample_un_mirror()``, encoded and published by the catalogue job's own writer
(``origin_catalogue.artifact``), and ``tests/origin_catalogue/test_fixture.py`` fails when the
committed files stop matching it.

``tests/fixtures/origin_catalogue_eval`` holds a build the catalogue job published from the
dev graph. ``--eval-from`` copies the published build of a local catalogue root there:
``current.json`` and the build directory it names, after the loader has accepted them under
the service's memory budget.

Each directory holds a format root's contents, which
``scripts/stage_origin_catalogue_fixture.py`` presents as a catalogue root.

Usage:
    uv run python scripts/record_origin_catalogue_fixture.py
    uv run python scripts/record_origin_catalogue_fixture.py --eval-from <catalogue-root>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from origin_catalogue.artifact import load_catalogue  # noqa: E402
from origin_catalogue.schema import FORMAT_ROOT, POINTER_FILE, VERSIONS_PREFIX  # noqa: E402
from origin_catalogue.service import DEFAULT_MAX_RESIDENT_BYTES  # noqa: E402
from scripts.stage_origin_catalogue_fixture import stage_fixture  # noqa: E402
from tests.origin_catalogue.support import (  # noqa: E402
    FIXTURE_BUILD_ID,
    FIXTURE_DIR,
    publish_sample,
    sample_un_mirror,
)

EVAL_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "origin_catalogue_eval"


def record_sample() -> None:
    # `publish_build` writes each build file once, so the old build goes first.
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    FIXTURE_DIR.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as root:
        publish_sample(
            stage_fixture(FIXTURE_DIR, Path(root)),
            build_id=FIXTURE_BUILD_ID,
            un_mirror=sample_un_mirror(),
        )
    print(f"wrote build {FIXTURE_BUILD_ID!r} to {FIXTURE_DIR}")


def record_eval(catalogue_root: Path) -> None:
    """Replaces the eval build with the one `catalogue_root` publishes, leaving the lease
    and state files the job keeps beside `current.json` behind."""
    snapshot = load_catalogue(
        catalogue_root.resolve().as_uri(), max_resident_bytes=DEFAULT_MAX_RESIDENT_BYTES
    )
    source = catalogue_root / FORMAT_ROOT
    versions = EVAL_FIXTURE_DIR / VERSIONS_PREFIX
    shutil.rmtree(versions, ignore_errors=True)
    versions.mkdir(parents=True)
    shutil.copytree(source / VERSIONS_PREFIX / snapshot.build_id, versions / snapshot.build_id)
    shutil.copyfile(source / POINTER_FILE, EVAL_FIXTURE_DIR / POINTER_FILE)
    print(f"wrote build {snapshot.build_id!r} to {EVAL_FIXTURE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--eval-from",
        type=Path,
        metavar="CATALOGUE_ROOT",
        help="a local catalogue root, the --local-dir of an origin_catalogue.job run",
    )
    args = parser.parse_args()
    if args.eval_from is None:
        record_sample()
    else:
        record_eval(args.eval_from)


if __name__ == "__main__":
    main()
