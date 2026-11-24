"""Post a query, then each part's base and entries plus `<axis>:whole`; print the pairs."""

import json
import sys
import urllib.request


def post(url: str, body: dict) -> dict:
    data = json.dumps({"schema_version": 1, **body}).encode()
    headers = {"content-type": "application/json"}
    request = urllib.request.Request(f"{url}/api/search/resolve", data, headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        return {"refused": json.load(error)}


def values(spec: dict) -> dict:
    totals: dict = {}
    for series in (spec["resolved"].get("observations") or {}).get("series", []):
        for point in (p for p in series["points"] if p.get("value") is not None):
            totals[point["date"]] = totals.get(point["date"], 0) + point["value"]
    return dict(sorted(totals.items()))


def whole_pair(url: str, query: str, axis: str) -> list[dict]:
    options = {"response_options": {"include_observations": True}}
    pairs = []
    for part in post(url, {"input": {"kind": "raw_text", "query": query}, **options})["parts"]:
        scope = part["spec"]["resolved"]["scope_choices"]
        origin = {"kind": "choice", "choice_id": f"{scope['id']}:{axis}"}
        entry = {"axis": axis, "option_id": f"{axis}:whole", "provenance": "explicit"}
        resubmit = {"entry_kind": "identity", "variable_text": f"the whole of {query}"}
        resubmit |= {"identity": scope["base"]["identity"]}
        resubmit |= {"scope_entries": [*scope["entries"], {**entry, "origin": origin}]}
        answer = post(
            url, {"input": {"kind": "dataset_resubmit", "entries": [resubmit]}, **options}
        )
        pair = {"part": part["spec"]["spec_id"], "part_values": values(part["spec"])}
        if "refused" not in answer:
            whole = answer["parts"][0]["spec"]
            pair |= {"whole": whole["spec_id"], "whole_values": values(whole)}
            pair["derived"] = whole["resolved"]["scope_choices"].get("derived")
        pairs.append(pair | {"refused": answer.get("refused")})
    return pairs


if __name__ == "__main__":
    print(json.dumps(whole_pair(*sys.argv[1:4]), indent=1))
