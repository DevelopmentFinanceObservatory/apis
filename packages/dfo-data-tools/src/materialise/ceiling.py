"""Checks the sync row ceiling after filtering and before `Frame.build`.
Generation also checks during fetch, except for `latest` windows. `pipeline._generate_all`
checks collection totals.
"""

from __future__ import annotations

from materialise.errors import OverCeilingError
from materialise.window import Coverage

DEFAULT_SYNC_CEILING_ROWS = 250_000
"""Generic sync row cap for callers without an explicit limit, measured through Datasets
spikes. `datasets_api` admission uses `series_coverage`, but passes a separate fixed ceiling
for the mid-fetch guard. That value is not published in capabilities. Other package
consumers inherit this default."""


def check_ceiling(coverage: Coverage, *, ceiling: int = DEFAULT_SYNC_CEILING_ROWS) -> None:
    """Raises `OverCeilingError` (422, `over_ceiling`) when rows exceed `ceiling`. It returns
    normally when rows are within the limit. `Coverage` supplies dimension counts for error
    detail."""
    if coverage.rows <= ceiling:
        return
    raise OverCeilingError(
        f"materialised coverage of {coverage.rows} rows exceeds the sync "
        f"materialisation ceiling of {ceiling} rows",
        detail=(
            f"materialised coverage of {coverage.rows} rows exceeds the sync "
            f"materialisation ceiling of {ceiling} rows (dimensions: "
            f"{coverage.dimensions!r})"
        ),
        ceiling_kind="rows",
        count=coverage.rows,
        cap=ceiling,
    )
