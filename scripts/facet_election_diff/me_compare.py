"""Before/after table for measurement 2: counts by disagreement kind, from before/ and after/."""

import collections
import glob
import json
from pathlib import Path

W = {
    "2019-2023": (2019, 2023),
    "1995-2000": (1995, 2000),
    "latest": None,
    "2022-2027": (2022, 2027),
}


def counts(d, extra):
    files = [f for f in glob.glob(f"{d}/m24_results_*.jsonl") if ("2022-2027" in f) == extra]
    rows = [json.loads(line) for f in files for line in Path(f).read_text().splitlines()]
    c = collections.Counter()
    ex = collections.defaultdict(list)
    for r in rows:
        if "error" in r:
            c["error"] += 1
        if "mint_refused" in r:
            c["mint_refused"] += 1
        if "pair" not in r:
            continue
        n = "one" if r["confirmed"] == 1 else "multi"
        c["evaluated"] += 1
        for comp in ("pair_vs_mint", "bulk_vs_mint", "pair_vs_bulk"):
            d_ = r.get(comp)
            if not d_:
                continue
            a, b = comp.split("_vs_")
            if d_["lead"]:
                c[(comp, f"lead, {n}")] += 1
                ex[(comp, "lead")].append(r)
            for k in ("unserved", "continuation"):
                if d_[k]:
                    c[(comp, k)] += 1
                    ex[(comp, k)].append(r)
            if d_["members"]:
                sv = d_["members"][0]
                diff = set(r[a]["selected"].get(sv, [])) ^ set(r[b]["selected"].get(sv, []))
                facets = {f[0]: f for f in r["facets"].get(sv, [])}
                w = W[r["window"]]

                def where(fid, facets=facets, w=w):
                    f = facets.get(fid)
                    if f is None:
                        return "unknown"
                    if w is None:
                        return "latest"
                    return "in" if int(f[2]) <= w[1] and int(f[3]) >= w[0] else "out"

                wh = "/".join(sorted({where(x) for x in diff}))
                kind = f"members, {n}, " + {
                    "in": "second facet in window",
                    "out": "facet outside window",
                    "latest": "extra facet under latest",
                }.get(wh, wh)
                c[(comp, kind)] += 1
                ex[(comp, kind)].append(r)
    return c, ex


if __name__ == "__main__":
    for extra in (False, True):
        b, _ = counts("before", extra)
        a, ex = counts("after", extra)
        print("== extra window 2022-2027" if extra else "== three windows")
        print(
            "evaluated",
            b["evaluated"],
            a["evaluated"],
            "errors",
            b["error"],
            a["error"],
            "refused",
            b["mint_refused"],
            a["mint_refused"],
        )
        keys = sorted({k for k in b | a if isinstance(k, tuple)}, key=lambda k: (k[1], k[0]))
        for k in keys:
            print(f"{k[1]:55} {k[0]:13} before {b[k]:5} after {a[k]:5}")
