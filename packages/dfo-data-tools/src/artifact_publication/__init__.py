"""Storage and publication shared by the jobs that publish versioned
artifacts: the coverage index (`series_coverage.job`) and the origin
catalogue (`origin_catalogue.job`).

Each artifact lives under a format root, where `POINTER_FILE` names one
immutable build directory under `VERSIONS_PREFIX`. `uri` reads and writes
`gs://` and `file://` objects; `jobs` holds the BigQuery reads, build lease,
pruning and stale-publish tracking both jobs run.
"""

from __future__ import annotations

from typing import Final

POINTER_FILE: Final[str] = "current.json"
"""The published pointer at a format root. A publish writes it last, so it
is the commit point."""

VERSIONS_PREFIX: Final[str] = "versions"
"""The sub-prefix of a format root holding one directory per build,
`versions/<build_id>/`, each written once by the run that created it."""

__all__ = ["POINTER_FILE", "VERSIONS_PREFIX"]
