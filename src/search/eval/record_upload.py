"""Copy run records to the eval runs bucket with create permission alone.

``python -m search.eval.record_upload --bucket BUCKET ROOT`` copies every run record under
``ROOT/apis-search/`` to ``gs://BUCKET/apis-search/<run id>/``, ``results.jsonl`` first and
``manifest.json`` second, since a reader treats a run with a manifest as complete.

The nightly's writer account may only create objects under ``apis-search/``, so each object
goes up with ``if_generation_match=0``: Cloud Storage creates it when no object of that name
exists and refuses otherwise, without the account reading anything. ``gcloud storage cp``
reads the destination before it writes, which that account may not do. An object the bucket
already holds is left as it is and reported, so running the upload again finishes a run whose
manifest did not go up.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Protocol

from dfo_evals.records import MANIFEST_FILE, RESULTS_FILE, iter_runs
from google.api_core.exceptions import PreconditionFailed

from search.eval.corpus_dataset import SUITE

# The generation precondition that matches only when no object of the name exists.
_ABSENT: Final = 0
# The order a run's files go up in: the manifest last, so a reader never sees a manifest
# without its results.
_UPLOAD_ORDER: Final = (RESULTS_FILE, MANIFEST_FILE)

logger = logging.getLogger(__name__)


class Blob(Protocol):
    """The part of ``google.cloud.storage.Blob`` the upload uses."""

    def upload_from_filename(self, filename: str, *, if_generation_match: int) -> None:
        """Upload a local file to the object."""


class Bucket(Protocol):
    """The part of ``google.cloud.storage.Bucket`` the upload uses."""

    def blob(self, blob_name: str) -> Blob:
        """Return a handle on an object, without reading it."""


def upload_runs(*, bucket: Bucket, root: Path) -> list[str]:
    """Copy every run record under ``root/apis-search/`` to the bucket.

    Args:
        bucket: The eval runs bucket.
        root: The records root the nightly wrote (``--records``).

    Returns:
        The object names created, in upload order.

    Raises:
        FileNotFoundError: ``root/apis-search`` does not exist.
    """
    created = []
    for run_dir in iter_runs(root=root, suite=SUITE):
        for name in _UPLOAD_ORDER:
            object_name = f"{SUITE}/{run_dir.name}/{name}"
            try:
                bucket.blob(object_name).upload_from_filename(
                    str(run_dir / name), if_generation_match=_ABSENT
                )
            except PreconditionFailed:
                logger.warning("gs://.../%s already exists; left as it is", object_name)
                continue
            created.append(object_name)
            logger.info("created %s", object_name)
    return created


def main(argv: Sequence[str] | None = None) -> int:
    """Upload the run records the arguments name and return 0; any error raises."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", required=True, help="the eval runs bucket's name")
    parser.add_argument("root", type=Path, help="the records root (the nightly's --records)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from google.cloud import storage

    bucket = storage.Client().bucket(args.bucket)
    created = upload_runs(bucket=bucket, root=args.root)
    logger.info("created %d objects in gs://%s", len(created), args.bucket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
