"""`python -m series_coverage.verify` checks that the serving image can load
its coverage artifact. It calls the same `load_coverage` function used at
startup and checks only this image's format root. Missing, malformed, or
incompatible artifacts print an error and return a non-zero status.

Pass an artifact URI as an argument, or set `COVERAGE_ARTIFACT_URI`.
A skipped or discarded job can exit 0 without publishing an artifact, so
deploys run this check before moving services to the image.

Usage:
    python -m series_coverage.verify                      # reads COVERAGE_ARTIFACT_URI
    python -m series_coverage.verify file:///tmp/coverage  # verifies an explicit URI
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from series_coverage.loader import load_coverage

_ARTIFACT_URI_ENV_VAR = "COVERAGE_ARTIFACT_URI"


class _VerifyEnv(BaseSettings):
    model_config = SettingsConfigDict()

    artifact_uri: str | None = Field(default=None, alias=_ARTIFACT_URI_ENV_VAR)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print(
            f"usage: python -m series_coverage.verify [artifact_uri] "
            f"(reads {_ARTIFACT_URI_ENV_VAR} when omitted)",
            file=sys.stderr,
        )
        return 2
    if args:
        artifact_uri = args[0]
    else:
        artifact_uri = _VerifyEnv().artifact_uri
        if not artifact_uri:
            print(
                f"series_coverage.verify: no artifact_uri given and {_ARTIFACT_URI_ENV_VAR} is "
                "not set -- nothing to verify",
                file=sys.stderr,
            )
            return 1
    try:
        loaded = load_coverage(artifact_uri)
    except Exception as exc:
        print(
            f"series_coverage.verify: {artifact_uri!r} did not load a compatible coverage "
            f"artifact: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        f"series_coverage.verify: {artifact_uri!r} loaded a compatible coverage artifact "
        f"(stamp {loaded.manifest.stamp!r}, build {loaded.manifest.build_id!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
