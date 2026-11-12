"""Enumerate the populations the measurements run over and write populations.json.

- largest: the three index families (dataset, five-tuple, property set) with the most
  coverage rows, every member, nothing bound: a bare bulk read of the family.
- siblings: groups of a family's rows that differ on one constraint property only (a
  breakdown answer), sampled, each open and bound to its most-held recipient.
- multimethod: every member holding facets under two or more methods, alone, open and
  bound to its most-held recipient and donor.
- corpus: corpus cases naming an index family (me_common.corpus_populations).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from me_common import corpus_populations, families
from me_load import HERE, load

loaded, views = load()
cov, ci = views.coverage, views.code_index
t = cov._table
rows_by_var = {
    r["variable"]: r["n_obs_count"]
    for r in t.group_by(["variable"]).aggregate([("n_obs", "count")]).to_pylist()
}
coverage_vars = set(rows_by_var)
meta = cov.facet_metadata
methods = defaultdict(set)
for r in t.group_by(["variable", "facet_id"]).aggregate([]).to_pylist():
    m = meta.get(r["facet_id"])
    methods[r["variable"]].add(m.measurement_method if m else None)


def top_party(svs, column):
    sub = t.filter(pc.is_in(t["variable"], pa.array(svs)))
    c = Counter(sub[column].to_pylist())
    c.pop("", None)
    return c.most_common(1)[0][0] if c else None


pops = []
fam_list = []
for ds, fam, members in families(ci, coverage_vars):
    fam_list.append((sum(rows_by_var[m] for m in members), ds, fam, members))
fam_list.sort(key=lambda x: -x[0])
print("families", len(fam_list))
for n, ds, fam, members in fam_list[:3]:
    print("largest", n, ds, len(members), fam.five_tuple)
    pops.append(
        {
            "name": f"largest:{ds}:{fam.five_tuple.measured_property}:{fam.five_tuple.population_type}",
            "members": members,
            "bound": {},
            "dataset": ds,
            "source": "largest",
            "rows": n,
        }
    )

rng = random.Random(20261112)
sib = []
for _n, ds, fam, members in fam_list:
    rows = [ci.statvar_by_dcid(m) for m in members]
    for prop in sorted(fam.properties):
        groups = defaultdict(list)
        for r in rows:
            if prop in r.constraint_properties:
                key = tuple(sorted((k, v) for k, v in r.constraint_properties.items() if k != prop))
                groups[key].append(r.dcid)
        sib.extend((ds, prop, svs) for svs in groups.values() if 2 <= len(svs) <= 40)
print("sibling groups", len(sib))
multi = {sv for sv, ms in methods.items() if len(ms) > 1}
chosen = [g for g in sib if multi & set(g[2])]
rest = [g for g in sib if not (multi & set(g[2]))]
chosen += rng.sample(rest, min(300, len(rest)))
for ds, prop, svs in chosen:
    name = f"siblings:{ds}:{prop}:{svs[0][:60]}"
    pops.append({"name": name, "members": svs, "bound": {}, "dataset": ds, "source": "siblings"})
    rec = top_party(svs, "recipient")
    if rec:
        pops.append(
            {
                "name": name + f"@recipient={rec}",
                "members": svs,
                "bound": {"recipient": [rec]},
                "dataset": ds,
                "source": "siblings",
            }
        )

for sv in sorted(multi):
    ds = ci.statvar_by_dcid(sv).provenances[0]
    pops.append(
        {
            "name": f"multimethod:{sv}",
            "members": [sv],
            "bound": {},
            "dataset": ds,
            "source": "multimethod",
        }
    )
    for col in ("recipient", "donor"):
        p = top_party([sv], col)
        if p:
            pops.append(
                {
                    "name": f"multimethod:{sv}@{col}={p}",
                    "members": [sv],
                    "bound": {col: [p]},
                    "dataset": ds,
                    "source": "multimethod",
                }
            )

pops.extend(
    {
        "name": p.name,
        "members": p.members,
        "bound": {k: sorted(v) for k, v in p.bound.items()},
        "dataset": p.dataset,
        "source": "corpus",
    }
    for p in corpus_populations(ci, coverage_vars, Path(__file__).resolve().parents[2] / "corpus")
)

print(Counter(p["source"] for p in pops))
(HERE / "populations.json").write_text(json.dumps(pops, indent=0))
