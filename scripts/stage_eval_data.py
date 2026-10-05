#!/usr/bin/env python3
"""Download the deployed coverage and origin catalogue builds for the paired eval.

Each ``--checkout`` names a dfo-apis tree with a synced environment. The script asks every
checkout, through its own environment, for ``series_coverage.schema.FORMAT_ROOT`` and
``origin_catalogue.schema.FORMAT_ROOT``, so a base and a head on different artifact formats
each get the root they read. For every distinct root it reads the deployed ``current.json``,
copies the one build that pointer names, and writes the pointer beside it. Both apps and
the runner then read the same bytes from ``--dest``.

It prints ``COVERAGE_ARTIFACT_URI=file://...`` and ``ORIGIN_CATALOGUE_URI=file://...``,
ready to append to ``$GITHUB_ENV``. The sources default to those two variables, so a shell
that already points at the deployed buckets stages them with no flags. It exits 3 when a
root has no pointer, or its pointer names a build with no objects.

Usage:
    uv run python scripts/stage_eval_data.py --checkout . --checkout ../base --dest staged
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from artifact_publication import POINTER_FILE, VERSIONS_PREFIX
from artifact_publication.uri import join_uri, list_uri_object_times, read_uri_bytes

COVERAGE_ENV = "COVERAGE_ARTIFACT_URI"
CATALOGUE_ENV = "ORIGIN_CATALOGUE_URI"

# Printed one per line, coverage first, in the order `checkout_format_roots` reads them.
_ROOTS_PROGRAM = (
    "from series_coverage.schema import FORMAT_ROOT as coverage\n"
    "from origin_catalogue.schema import FORMAT_ROOT as catalogue\n"
    "print(coverage)\n"
    "print(catalogue)\n"
)


@dataclass(frozen=True, slots=True)
class FormatRoots:
    """The format roots one checkout reads under each artifact base URI."""

    coverage: str
    catalogue: str


class MissingRootError(Exception):
    """A deployed root has no pointer, or its pointer names a build with no objects."""


def checkout_format_roots(checkout: Path) -> FormatRoots:
    """Ask ``checkout``'s own environment which format roots its code reads.

    Raises:
        subprocess.CalledProcessError: the checkout's environment cannot import the schemas.
    """
    out = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", _ROOTS_PROGRAM],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(out) != 2:
        raise ValueError(f"{checkout}: expected two format roots, got {out!r}")
    return FormatRoots(coverage=out[0], catalogue=out[1])


def stage_root(source_uri: str, root: str, dest: Path) -> str:
    """Copy the build ``<source_uri>/<root>/current.json`` names into ``dest/<root>``.

    Returns the staged build id.

    Raises:
        MissingRootError: the root has no pointer, or the build it names has no objects.
    """
    source_root = join_uri(source_uri, root)
    try:
        pointer = read_uri_bytes(join_uri(source_root, POINTER_FILE))
    except FileNotFoundError as exc:
        raise MissingRootError(f"no {POINTER_FILE} at {source_root}") from exc
    # Both artifacts' pointers name their build directory by `build_id`; the other fields
    # differ by artifact and stay as published.
    build_id = json.loads(pointer)["build_id"]
    build_uri = join_uri(source_root, VERSIONS_PREFIX, build_id)
    objects = sorted(list_uri_object_times(build_uri))
    if not objects:
        raise MissingRootError(f"{POINTER_FILE} at {source_root} names {build_id}, which is empty")
    build_dir = dest / root / VERSIONS_PREFIX / build_id
    for name in objects:
        target = build_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read_uri_bytes(join_uri(build_uri, name)))
    (dest / root / POINTER_FILE).write_bytes(pointer)
    return build_id


def stage(
    checkouts: Sequence[Path],
    *,
    coverage_uri: str,
    catalogue_uri: str,
    dest: Path,
    roots_of: Callable[[Path], FormatRoots],
) -> dict[str, str]:
    """Stage every root the checkouts read and return the environment that reads them.

    Coverage stages under ``dest/coverage`` and the catalogue under
    ``dest/origin-catalogue``. A root two checkouts share is copied once.

    Raises:
        MissingRootError: see ``stage_root``.
    """
    roots = [roots_of(checkout) for checkout in checkouts]
    targets = {
        COVERAGE_ENV: (coverage_uri, dest / "coverage", {r.coverage for r in roots}),
        CATALOGUE_ENV: (catalogue_uri, dest / "origin-catalogue", {r.catalogue for r in roots}),
    }
    env = {}
    for name, (source_uri, local, wanted) in targets.items():
        for root in sorted(wanted):
            build_id = stage_root(source_uri, root, local)
            print(f"staged {join_uri(source_uri, root)} build {build_id}", file=sys.stderr)
        env[name] = local.resolve().as_uri()
    return env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkout", type=Path, action="append", required=True, help="a tree to stage for"
    )
    parser.add_argument("--dest", type=Path, required=True, help="the local directory to fill")
    parser.add_argument(
        "--coverage-uri",
        default=os.environ.get(COVERAGE_ENV),
        help=f"the deployed coverage base URI (default: ${COVERAGE_ENV})",
    )
    parser.add_argument(
        "--catalogue-uri",
        default=os.environ.get(CATALOGUE_ENV),
        help=f"the deployed origin catalogue base URI (default: ${CATALOGUE_ENV})",
    )
    args = parser.parse_args(argv)
    if not (args.coverage_uri and args.catalogue_uri):
        parser.error(
            f"set --coverage-uri and --catalogue-uri, or {COVERAGE_ENV} and {CATALOGUE_ENV}"
        )
    try:
        env = stage(
            args.checkout,
            coverage_uri=args.coverage_uri,
            catalogue_uri=args.catalogue_uri,
            dest=args.dest,
            roots_of=checkout_format_roots,
        )
    except MissingRootError as exc:
        print(f"stage_eval_data: {exc}", file=sys.stderr)
        return 3
    for name, uri in env.items():
        print(f"{name}={uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
