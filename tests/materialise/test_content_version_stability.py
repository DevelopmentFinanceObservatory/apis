"""Pins a content-version token for a fixed frame and lineage without conversion.

The test runs against whatever `dfo-spec-kit-frame-grammar` revision `pyproject.toml` pins.
It does not prove the conversion-recipe identity rules leave unconverted frames unchanged;
`dfo-spec-kit`'s vector suite does. A pin bump that moves the digest is an identity rotation
to trace, not a literal to update.
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
    """Pins the digest for graph revision `graph-rev-2024-01`, source
    `OECD_DAC_CRS` revision `rev-7`, and bound provenance `OECD_DAC2A`.
    """
    token = mint_version(
        _FRAME,
        graph_revision="graph-rev-2024-01",
        source_revisions=[{"source": "OECD_DAC_CRS", "revision": "rev-7"}],
        facets=[{"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"}],
    )

    assert token == "cv_ea9dc1d4c4be9cf2ad62a810aa2f4db7"
