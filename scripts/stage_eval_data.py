#!/usr/bin/env python3
"""Download the deployed graph release and origin catalogue build for the paired eval.

Each ``--checkout`` names a dfo-apis tree with a synced environment. The script asks every
checkout, through its own environment, for ``origin_catalogue.schema.FORMAT_ROOT``, so a base
and a head on different artifact formats each get the root they read. For every distinct
root it reads the deployed ``current.json``, copies the one build that pointer names, and
writes the pointer beside it. It copies the graph release the deployed pointer names, in this
tree's release format, with its manifest and parts. Both apps and the runner then read the
same bytes from ``--dest``.

It prints ``GRAPH_RELEASE_URI=file://...`` and ``ORIGIN_CATALOGUE_URI=file://...``, ready to
append to ``$GITHUB_ENV``. The sources default to those two variables, so a shell that already
points at the deployed buckets stages them with no flags. It exits 3 when a root or the
release has no pointer, or a pointer names a build with no objects.

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
from pathlib import Path

from artifact_publication import POINTER_FILE, VERSIONS_PREFIX
from artifact_publication.uri import (
    download_uri_to_path,
    join_uri,
    list_uri_object_times,
    read_uri_bytes,
)
from graph_release.context import URI_ENV_VAR as RELEASE_ENV
from graph_release.format import (
    MANIFESTS_DIR,
    PARTS_DIR,
    POINTER_PATH,
    Manifest,
    manifest_uri,
    part_uri,
    pointer_uri,
    read_pointer,
)

CATALOGUE_ENV = "ORIGIN_CATALOGUE_URI"

_ROOT_PROGRAM = "from origin_catalogue.schema import FORMAT_ROOT\nprint(FORMAT_ROOT)\n"


class MissingRootError(Exception):
    """A deployed root or the release has no pointer, or a pointer names a build with no
    objects."""


def checkout_catalogue_root(checkout: Path) -> str:
    """Ask ``checkout``'s own environment which origin catalogue format root its code reads.

    Raises:
        subprocess.CalledProcessError: the checkout's environment cannot import the schema.
    """
    return subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", _ROOT_PROGRAM],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    # The pointer names its build directory by `build_id`; its other fields stay as published.
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


def stage_release(source_uri: str, dest: Path) -> str:
    """Copy the graph release ``<source_uri>``'s pointer names, its manifest and every part
    the manifest lists into ``dest``, and the pointer beside them.

    Returns the staged release id.

    Raises:
        MissingRootError: no release is published under ``source_uri``.
    """
    pointer = read_pointer(source_uri)
    if pointer is None:
        raise MissingRootError(f"no release pointer at {pointer_uri(source_uri)}")
    manifest = read_uri_bytes(manifest_uri(source_uri, pointer.release_id))
    (dest / PARTS_DIR).mkdir(parents=True, exist_ok=True)
    for ref in Manifest.from_json(manifest).parts:
        download_uri_to_path(part_uri(source_uri, ref), dest / PARTS_DIR / ref.file_name)
    (dest / MANIFESTS_DIR).mkdir(parents=True, exist_ok=True)
    (dest / MANIFESTS_DIR / f"{pointer.release_id}.json").write_bytes(manifest)
    (dest / POINTER_PATH).parent.mkdir(parents=True, exist_ok=True)
    (dest / POINTER_PATH).write_bytes(pointer.to_json())
    return pointer.release_id


def stage(
    checkouts: Sequence[Path],
    *,
    release_uri: str,
    catalogue_uri: str,
    dest: Path,
    root_of: Callable[[Path], str],
) -> dict[str, str]:
    """Stage the graph release and every catalogue root the checkouts read, and return the
    environment that reads them.

    The release stages under ``dest/graph-release`` and the catalogue under
    ``dest/origin-catalogue``. A root two checkouts share is copied once.

    Raises:
        MissingRootError: see ``stage_root`` and ``stage_release``.
    """
    roots = sorted({root_of(checkout) for checkout in checkouts})
    release_dir = dest / "graph-release"
    release_id = stage_release(release_uri, release_dir)
    print(f"staged {release_uri} release {release_id}", file=sys.stderr)
    env = {RELEASE_ENV: release_dir.resolve().as_uri()}

    catalogue_dir = dest / "origin-catalogue"
    for root in roots:
        build_id = stage_root(catalogue_uri, root, catalogue_dir)
        print(f"staged {join_uri(catalogue_uri, root)} build {build_id}", file=sys.stderr)
    env[CATALOGUE_ENV] = catalogue_dir.resolve().as_uri()
    return env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkout", type=Path, action="append", required=True, help="a tree to stage for"
    )
    parser.add_argument("--dest", type=Path, required=True, help="the local directory to fill")
    parser.add_argument(
        "--release-uri",
        default=os.environ.get(RELEASE_ENV),
        help=f"the deployed graph release URI (default: ${RELEASE_ENV})",
    )
    parser.add_argument(
        "--catalogue-uri",
        default=os.environ.get(CATALOGUE_ENV),
        help=f"the deployed origin catalogue base URI (default: ${CATALOGUE_ENV})",
    )
    args = parser.parse_args(argv)
    if not (args.release_uri and args.catalogue_uri):
        parser.error(f"set --release-uri and --catalogue-uri, or {RELEASE_ENV} and {CATALOGUE_ENV}")
    try:
        env = stage(
            args.checkout,
            release_uri=args.release_uri,
            catalogue_uri=args.catalogue_uri,
            dest=args.dest,
            root_of=checkout_catalogue_root,
        )
    except MissingRootError as exc:
        print(f"stage_eval_data: {exc}", file=sys.stderr)
        return 3
    for name, uri in env.items():
        print(f"{name}={uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
