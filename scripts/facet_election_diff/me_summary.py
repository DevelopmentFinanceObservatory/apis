import collections
import glob
import json
from pathlib import Path

rows = []
for f in sorted(glob.glob("m24_results_*.jsonl")):
    rows += [json.loads(line) for line in Path(f).read_text().splitlines()]
print(
    "records",
    len(rows),
    "errors",
    sum("error" in r for r in rows),
    "mint refused",
    sum("mint_refused" in r for r in rows),
)
for r in rows:
    if "error" in r:
        print(r["population"], r["error"])
        break
c = collections.defaultdict(collections.Counter)
for r in rows:
    if "pair" not in r:
        c[r["source"]]["no_member_in_window"] += 1
        continue
    kind = "one" if r["confirmed"] == 1 else "multi"
    c[r["source"]]["evaluated_" + kind] += 1
    for k in ("pair_vs_mint", "bulk_vs_mint", "pair_vs_bulk"):
        d = r.get(k)
        if not d:
            continue
        for kk in ("lead", "unserved", "continuation"):
            if d[kk]:
                c[r["source"]][f"{k}:{kk}:{kind}"] += 1
        if d["members"]:
            c[r["source"]][f"{k}:members:{kind}"] += 1
    m4 = r.get("m4")
    if m4:
        for kn, v in m4.items():
            c[r["source"]][f"m4:{kn}:evaluated"] += 1
            if v["keys_differ"]:
                c[r["source"]][f"m4:{kn}:keys_differ"] += 1
            if v["lead_differs"]:
                c[r["source"]][f"m4:{kn}:lead_differs"] += 1
for s, cc in c.items():
    print(s)
    [print("   ", k, v) for k, v in sorted(cc.items())]
