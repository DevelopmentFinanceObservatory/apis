"""Compare binding against the worked line-chart fixture.

The test checks the exact points, label, and source reference for the matching observation window.
It skips the worked series' structured `unit`, which binding cannot produce, and the fixture's
select dimension names and series id, which are illustrative placeholders.
"""

from __future__ import annotations

import json

from assembler.bind import bind_all
from assembler.resolve import resolve_selectors
from assembler.testing import malaria_ethiopia_resolved_frame


def test_bound_points_match_the_worked_line_chart_fixture(catalogue_dir):
    payload_path = catalogue_dir / "fixtures" / "line_chart" / "payload.json"
    worked_payload = json.loads(payload_path.read_text())
    worked_series = worked_payload["component"]["series"][0]
    worked_window = worked_payload["component"]["x_axis"]["window"]

    resolved_frame = malaria_ethiopia_resolved_frame()
    selectors = [
        {
            "bindings": [
                {"dimension": "stat_var", "values": ["dcid:MalariaGrantDisbursements"]},
                {"dimension": "recipient", "values": ["country/ETH"]},
            ]
        }
    ]
    series = resolve_selectors(
        resolved_frame=resolved_frame, selectors=selectors, window=worked_window
    )
    assert len(series) == 1

    (bound,) = bind_all(resolved_frame=resolved_frame, series=series, window=worked_window)

    assert list(bound.points) == worked_series["points"]
    assert bound.label == worked_series["label"]
    assert bound.source_ref == worked_series["source_ref"]
