"""Ask Search for a part and its whole over one axis, and print the pair.

    uv run python scripts/whole_pair.py URL "United States bilateral ODA 2015-2023" provider

Posts the query, then each definite part's own base and entries plus one entry,
`<axis>:whole`, back as a dataset_resubmit, and prints the `derived` block and each
figure's values by year. The client edits no identity.
"""

import json
import sys
import urllib.error
import urllib.request

OPTIONS = {"include_observations": True}


def post(url: str, body: dict) -> dict:
    data = json.dumps({"schema_version": 1, "response_options": OPTIONS, **body}).encode()
    request = urllib.request.Request(
        f"{url}/api/search/resolve", data, {"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        return {"refused": json.load(error)}


def values(spec: dict) -> dict:
    totals: dict = {}
    for series in ((spec.get("resolved") or {}).get("observations") or {}).get("series", []):
        for point in series["points"]:
            if point.get("value") is not None:
                totals[point["date"]] = totals.get(point["date"], 0) + point["value"]
    return dict(sorted(totals.items()))


def whole_pair(url: str, query: str, axis: str) -> list[dict]:
    pairs = []
    for part in post(url, {"input": {"kind": "raw_text", "query": query}})["parts"]:
        if part.get("status") != "definite":
            continue
        scope = part["spec"]["resolved"]["scope_choices"]
        entry = {
            "axis": axis,
            "option_id": f"{axis}:whole",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": f"{scope['id']}:{axis}"},
        }
        whole = post(
            url,
            {
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": f"{axis} whole of {query}",
                            "identity": scope["base"]["identity"],
                            "scope_entries": [*scope["entries"], entry],
                        }
                    ],
                }
            },
        )
        if "refused" in whole:
            pairs.append({"part": part["spec"]["spec_id"], "refused": whole["refused"]})
            continue
        spec = whole["parts"][0]["spec"]
        pairs.append(
            {
                "part": part["spec"]["spec_id"],
                "whole": spec["spec_id"],
                "derived": spec["resolved"]["scope_choices"].get("derived"),
                "part_values": values(part["spec"]),
                "whole_values": values(spec),
            }
        )
    return pairs


if __name__ == "__main__":
    print(json.dumps(whole_pair(*sys.argv[1:4]), indent=1))
