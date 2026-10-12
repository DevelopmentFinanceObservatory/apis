"""Pins a content-version token for a fixed frame.

The test runs against whatever `dfo-spec-kit-frame-grammar` revision `pyproject.toml` pins.
The token hashes the canonical frame alone, so it moves only when the frame's bytes or the
grammar's preimage family change. A pin bump that moves the digest is an identity rotation to
trace, not a literal to update.
"""

from __future__ import annotations

from materialise.content_version import mint_version
from materialise.frame_build import build_frame

_FRAME = build_frame(
    role_keys=("provider", "recipient"),
    rows=[
        {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "provider": "country/FRA",
            "recipient": "country/TGO",
            "time_period": "2020",
            "value": 1000.0,
        }
    ],
)


def test_mint_version_digest_is_stable():
    assert mint_version(_FRAME) == "cv_2256683cb97b65989938bf1d3724279b"
