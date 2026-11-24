"""The figures the whole-option replay is checked against, read from the OECD's own tables.

    python3 scripts/whole_pair_oracle.py us_bilateral|g7_africa|g7_africa_inclusive

Reads the OECD SDMX API (DAC1, DAC2a) and prints each year's part, whole and share.
"""

import csv
import io
import json
import subprocess
import sys

API = "https://sdmx.oecd.org/public/rest/data/OECD.DCD.FSD,{flow},/{key}?startPeriod={start}&endPeriod={end}&format=csvfilewithlabels"
G7 = "G7"


def read(flow: str, key: str, start: int, end: int) -> list[dict]:
    url = API.format(flow=flow, key=key, start=start, end=end)
    # The API refuses urllib's default client; curl's is accepted.
    text = subprocess.run(["curl", "-sf", "-m", "300", url], check=True, capture_output=True).stdout
    return list(csv.DictReader(io.StringIO(text.decode())))


def summed(rows: list[dict], **match: str) -> dict[int, float]:
    totals: dict[int, float] = {}
    for row in rows:
        if all(row[k] == v for k, v in match.items()) and row["OBS_VALUE"]:
            year = int(row["TIME_PERIOD"])
            totals[year] = totals.get(year, 0.0) + float(row["OBS_VALUE"]) * 1e6
    return totals


def us_bilateral() -> dict:
    """US bilateral ODA over US total ODA, DAC1, OECD headline (net to 2017, GE after)."""
    rows = read("DSD_DAC1@DF_DAC1", "USA._Z.1010+1015._Z.1140+1160.USD.V", 2015, 2023)
    flow = {y: "1140" if y <= 2017 else "1160" for y in range(2015, 2024)}
    part = {y: summed(rows, MEASURE="1015", FLOW_TYPE=f)[y] for y, f in flow.items()}
    whole = {y: summed(rows, MEASURE="1010", FLOW_TYPE=f)[y] for y, f in flow.items()}
    return {"part": part, "whole": whole}


def g7_africa(measures: tuple[str, ...]) -> dict:
    """G7 ODA to Africa over G7 ODA to developing countries, DAC2a, `measures` summed."""
    rows = read("DSD_DAC2@DF_DAC2A", f"{G7}.F+DPGC.{'+'.join(measures)}.USD.V", 1960, 2023)
    side: dict[str, dict[int, float]] = {}
    for recipient in ("F", "DPGC"):
        for measure in measures:
            for year, value in summed(rows, RECIPIENT=recipient, MEASURE=measure).items():
                held = side.setdefault(recipient, {})
                held[year] = held.get(year, 0.0) + value
    years = {
        y for y in side["F"] if all(summed(rows, RECIPIENT="F", MEASURE=m).get(y) for m in measures)
    }
    return {"part": {y: side["F"][y] for y in sorted(years)}, "whole": side["DPGC"]}


if __name__ == "__main__":
    case = sys.argv[1]
    result = {
        "us_bilateral": us_bilateral,
        "g7_africa": lambda: g7_africa(("206",)),
        "g7_africa_inclusive": lambda: g7_africa(("206", "106")),
    }[case]()
    share = {
        y: round(v / result["whole"][y] * 100, 1)
        for y, v in result["part"].items()
        if result["whole"].get(y)
    }
    print(json.dumps({**result, "share": share}, indent=1))
