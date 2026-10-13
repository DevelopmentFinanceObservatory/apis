"""Writes `stored-canonical-frame-959f8bc.parquet` and `stored-canonical-frame-959f8bc.json`,
a canonical frame as dfo-apis 0f4b6e04e stored it under frame grammar kit 959f8bc, whose
`unit` struct is `struct<ref, request<currency, prices, base_year>, composed>`, a `request`
with no `recipe` member.

Run it with the interpreter of a dfo-apis checkout at 0f4b6e04e (kit 959f8bc), so the frame
is built and written by that release's own `materialise.frame_build.build_frame` and
`materialise.formats.write_canonical_frame`:

    PYTHONDONTWRITEBYTECODE=1 <checkout>/.venv/bin/python \
        tests/materialise/fixtures/make_stored_canonical_frame.py tests/materialise/fixtures

The JSON file holds that release's own reading of the frame: its `frame_sha256` and its
canonical (JCS) bytes, as `rows`, the oracle `test_stored_canonical_frame.py` reads the
stored frame against. The script refuses under any other unit layout.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow as pa
from dfo_frame_grammar import canonical_frame_bytes, canonical_frame_schema, frame_sha256

from materialise.formats import write_canonical_frame
from materialise.frame_build import build_frame

STEM = "stored-canonical-frame-959f8bc"

_KIT_959F8BC_UNIT = pa.struct(
    [
        pa.field("ref", pa.string()),
        pa.field(
            "request",
            pa.struct(
                [
                    pa.field("currency", pa.string()),
                    pa.field("prices", pa.string()),
                    pa.field("base_year", pa.int64()),
                ]
            ),
        ),
        pa.field("composed", pa.string()),
    ]
)

_UNITS: tuple[object, ...] = (
    "USDollar",
    {"currency": "USD", "prices": "constant", "base_year": 2021},
    {"currency": "EUR", "prices": "current"},
    {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
    {
        "numerator": {"currency": "USD", "prices": "constant", "base_year": 2021},
        "denominator": {"stat_var": "dcid:Count_Person"},
    },
    None,
)
"""A plain unit, two converted units, two composed units (one with a converted side) and no
unit, one per recipient."""


def main(out_dir: Path) -> None:
    schema = canonical_frame_schema(("stat_var", "time_period", "value", "unit", "status"))
    if schema.field("unit").type != _KIT_959F8BC_UNIT:
        raise SystemExit(f"not the kit 959f8bc unit layout: {schema.field('unit').type}")
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:Amount_FinancialFlow",
                "recipient": f"country/R{index}",
                "time_period": str(2020 + year),
                "value": index * 10.0 + year + 0.5,
                "unit": unit,
                "provenance": "OECD_CRS",
                "measurementMethod": "GrantEquivalent" if index % 2 else None,
            }
            for index, unit in enumerate(_UNITS)
            for year in range(2)
        ],
    )
    sink = io.BytesIO()
    write_canonical_frame((frame,), sink=sink)
    (out_dir / f"{STEM}.parquet").write_bytes(sink.getvalue())
    oracle = {
        "frame_sha256": frame_sha256(frame),
        "document": json.loads(canonical_frame_bytes(frame)),
    }
    (out_dir / f"{STEM}.json").write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
