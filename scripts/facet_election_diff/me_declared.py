"""Read the defaultMeasurementMethod each candidate variable declares, from the live dev graph."""

import collections
import json
import os
import time
from pathlib import Path

from me_load import load

from dc_graph.client import GraphClient

loaded, views = load()
cov = views.coverage
meta = cov.facet_metadata
vf = cov._table.group_by(["variable", "facet_id"]).aggregate([]).to_pylist()
methods = collections.defaultdict(set)
provs = collections.defaultdict(set)
for r in vf:
    m = meta.get(r["facet_id"])
    methods[r["variable"]].add(m.measurement_method if m else None)
    provs[r["variable"]].add(m.provenance_id if m else None)
want = sorted(
    v for v in methods if len(methods[v]) > 1 or "provenance/ONEClimateFinance" in provs[v]
)
print("variables to read", len(want))
g = GraphClient(
    base_url=os.environ["SEARCH_GRAPH_BASE"].rstrip("/") + "/core/api",
    token=os.environ.get("SEARCH_GRAPH_TOKEN") or None,
)
t = time.perf_counter()
out = {}
for i in range(0, len(want), 200):
    d = g.default_facet_policies(want[i : i + 200])
    out.update({sv: {ax: p.resolved_ref for ax, p in axes.items()} for sv, axes in d.items()})
print("read s", round(time.perf_counter() - t, 1), "declaring", len(out))
print(collections.Counter(v["measurementMethod"] for v in out.values()))
Path("declared.json").write_text(json.dumps({"read": want, "declared": out}, indent=1))
