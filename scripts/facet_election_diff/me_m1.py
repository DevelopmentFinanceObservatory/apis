"""Measurement 1: wall time and memory of the bulk election today against member probes plus
the group election, on one of the three largest populations. One stage set per process:

    run.sh me_m1.py <population index 0-2> <bulk|probe> <window name>
"""

from __future__ import annotations

import gc
import json
import sys

from me_common import (
    WINDOWS,
    PeakSampler,
    Population,
    bulk_election,
    confirmed_members,
    mint_election,
    pair_election,
    probes_for,
)
from me_load import HERE, load

i, mode, wname = int(sys.argv[1]), sys.argv[2], sys.argv[3]
w = next(x for x in WINDOWS if x.name == wname)
raw = [p for p in json.loads((HERE / "populations.json").read_text()) if p["source"] == "largest"][
    i
]
pop = Population(raw["name"], raw["members"], {}, raw["dataset"], "largest")

with PeakSampler() as s_load:
    loaded, views = load()
cov, ci = views.coverage, views.code_index
gc.collect()
out = {
    "population": pop.name,
    "members": len(pop.members),
    "rows": raw["rows"],
    "window": w.name,
    "mode": mode,
    "load": s_load.report(),
}

with PeakSampler() as s:
    members, rows = confirmed_members(cov, pop, w)
out["pair_rows"] = {**s.report(), "confirmed": len(members), "window_rows": rows.num_rows}

if mode == "bulk":
    with PeakSampler() as s:
        e = bulk_election(cov, ci, pop, w, members, rows)
    out["bulk_election"] = {**s.report(), "lead": e["lead"], "unserved": len(e["unserved"])}
else:
    with PeakSampler() as s:
        probes, narrowed = probes_for(cov, ci, pop, members)
    out["member_probes"] = {
        **s.report(),
        "candidates": sum(len(p.candidates) for p in probes.values()),
    }
    with PeakSampler() as s:
        e = pair_election(narrowed, pop, w, members)
    out["pair_group_election"] = {**s.report(), "lead": e["lead"], "unserved": len(e["unserved"])}
    with PeakSampler() as s:
        m = mint_election(probes, pop, w, members, ci)
    out["mint_group_election"] = {**s.report(), "lead": m["lead"], "unserved": len(m["unserved"])}
print(json.dumps(out))
with open(HERE / "m1_results.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
