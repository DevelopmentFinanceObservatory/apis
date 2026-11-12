import collections
import glob
import json
from pathlib import Path

W = {"2019-2023": (2019, 2023), "1995-2000": (1995, 2000), "latest": None}
rows = []
for f in sorted(glob.glob("m24_results_*.jsonl")):
    rows += [json.loads(line) for line in Path(f).read_text().splitlines()]
c = collections.Counter()
ex = collections.defaultdict(list)
tot = collections.Counter()
for r in rows:
    if "pair" not in r:
        continue
    kind_m = "one" if r["confirmed"] == 1 else "multi"
    tot[kind_m] += 1
    ms = {str(f[1]) for fs in r["facets"].values() for f in fs}
    fam = (
        "climate"
        if any("Marker" in m for m in ms)
        else "ids_estimated"
        if "Estimated" in ms
        else "other"
    )
    for comp in ("pair_vs_mint", "bulk_vs_mint", "pair_vs_bulk"):
        d = r.get(comp)
        if not d:
            continue
        a, b = comp.split("_vs_")
        if d["lead"]:
            c[(comp, "lead", kind_m, fam)] += 1
            if len(ex[(comp, "lead")]) < 4:
                ex[(comp, "lead")].append(
                    (r["population"][:130], r["window"], r[a]["lead"], r[b]["lead"])
                )
        if d["members"]:
            sv = d["members"][0]
            diff = set(r[a]["selected"].get(sv, [])) ^ set(r[b]["selected"].get(sv, []))
            facets = {f[0]: f for f in r["facets"].get(sv, [])}
            w = W[r["window"]]

            def inwin(fid, facets=facets, w=w):
                f = facets.get(fid)
                if f is None:
                    return "unknown"
                if w is None:
                    return "latest"
                return "in" if int(f[2]) <= w[1] and int(f[3]) >= w[0] else "out"

            where = "/".join(sorted({inwin(x) for x in diff}))
            c[(comp, "members", kind_m, fam, where)] += 1
            if len(ex[(comp, "members", where)]) < 3:
                ex[(comp, "members", where)].append(
                    (
                        r["population"][:130],
                        r["window"],
                        sv[:80],
                        r[a]["selected"].get(sv),
                        r[b]["selected"].get(sv),
                        [facets.get(x) for x in diff],
                    )
                )
        for k in ("unserved", "continuation"):
            if d[k]:
                c[(comp, k, kind_m, fam)] += 1
print(tot)
for k, v in sorted(c.items()):
    print(v, k)
for k, v in ex.items():
    print(k)
    for e in v:
        print("   ", e)
