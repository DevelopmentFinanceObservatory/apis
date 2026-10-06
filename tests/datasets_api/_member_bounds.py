"""The bounds a single mint's member walk runs under, for tests that resolve members
directly: the default subject ceiling and member-read deadline, counted from the call."""

from __future__ import annotations

from datasets_api.adapter import MemberReadBounds, member_read_bounds
from datasets_api.config import DatasetsBuildSettings


def mint_bounds() -> MemberReadBounds:
    """`member_read_bounds` under default settings with no request deadline, as a mint
    door computes them for a request that sets none."""
    return member_read_bounds(DatasetsBuildSettings(), request_deadline=None)
