"""Measurement 3: members holding two or more measurement methods with points in the window
under one (provenance, unit) agreement key, as an index-sourced group with no declared
default serves them today (every method). Splits each member's in-window series-years into
those two methods both hold (overlap) and those one method holds alone (partition), and
prices "one method per member": the series and series-years only a losing method holds,
where the member's method is the mint's member election (`_select_member_facets` over
`coverage_probe`, nothing bound) plus the method it names as continuing the series.

Also measurement 4, exhaustively over every sibling group and family: groups whose most-held
agreement keys differ between lifetime facets and facets with a point in the window.

Writes m3_results.json and m4_exhaustive.json."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from me_common import families
from me_load import HERE, load

from materialise import MaterialisationRequest
from materialise.facets import ElectedDataset, coverage_probe
from materialise.generation import _select_member_facets
from materialise.window import RequestedWindow
from tests.materialise.conftest import FakeGraphReader

loaded, views = load()
cov, ci = views.coverage, views.code_index
t = cov._table
meta = cov.facet_metadata
epoch = cov.horizon.epoch_year
RANGES = (("2019-2023", 2019, 2023), ("1995-2000", 1995, 2000), ("all", None, None))


def years_of(m0: int, m1: int) -> list[int]:
    out = []
    for w, v in ((0, m0), (1, m1)):
        base = epoch + 64 * w
        while v:
            low = v & -v
            out.append(base + low.bit_length() - 1)
            v ^= low
    return out


def var_rows(sv):
    sub = t.filter(pc.equal(t["variable"], pa.scalar(sv, t["variable"].type.value_type)))
    return list(
        zip(
            sub["facet_id"].to_pylist(),
            sub["donor"].to_pylist(),
            sub["recipient"].to_pylist(),
            sub["year_mask_0"].to_pylist(),
            sub["year_mask_1"].to_pylist(),
            strict=False,
        )
    )


methods = defaultdict(set)
for r in t.group_by(["variable", "facet_id"]).aggregate([]).to_pylist():
    m = meta.get(r["facet_id"])
    methods[r["variable"]].add(m.measurement_method if m else None)
multi = sorted(sv for sv, ms in methods.items() if len(ms) > 1)
graph = FakeGraphReader()

results = []
for sv in multi:
    rows = var_rows(sv)
    row = ci.statvar_by_dcid(sv)
    probe = coverage_probe(stat_var=sv, bound={}, coverage=cov)
    for wname, first, last in RANGES:
        window = (
            RequestedWindow(kind="range", start_year=first, end_year=last)
            if first
            else RequestedWindow(kind="all")
        )
        req = MaterialisationRequest(
            stat_var=sv,
            role_bindings={},
            declared_roles=(),
            facet_bindings={},
            window=window,
            index_sourced=True,
            dataset=ElectedDataset(
                provenance=row.provenances[0], member_provenances=row.provenances
            ),
        )
        el = _select_member_facets(
            req, probe.candidates, graph=graph, breadth=probe.breadth_in(window)
        )
        msel = el.selections["measurementMethod"]
        served = {msel.ref} | ({msel.continued_by} if msel.continued_by else set())
        by_key = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )  # key -> series -> year -> methods
        for fid, donor, rec, m0, m1 in rows:
            fm = meta.get(fid)
            key = (fm.provenance_id, fm.unit) if fm else None
            for y in years_of(m0, m1):
                if first is None or first <= y <= last:
                    by_key[key][(donor, rec)][y].add(fm.measurement_method if fm else None)
        for key, series in by_key.items():
            held = {m for ys in series.values() for ms in ys.values() for m in ms}
            if len(held) < 2:
                continue
            pts = sum(len(ys) for ys in series.values())
            overlap = sum(1 for ys in series.values() for ms in ys.values() if len(ms) > 1)
            series_multi = sum(
                1 for ys in series.values() if len({m for ms in ys.values() for m in ms}) > 1
            )
            lost_pts = sum(1 for ys in series.values() for ms in ys.values() if not (ms & served))
            lost_series = sum(
                1 for ys in series.values() if not any(ms & served for ms in ys.values())
            )
            results.append(
                {
                    "stat_var": sv,
                    "window": wname,
                    "key": list(key) if key else None,
                    "methods": sorted(map(str, held)),
                    "winner": msel.ref,
                    "continued_by": msel.continued_by,
                    "series": len(series),
                    "series_years": pts,
                    "overlap_series_years": overlap,
                    "series_holding_two": series_multi,
                    "lost_series": lost_series,
                    "lost_series_years": lost_pts,
                }
            )
(HERE / "m3_results.json").write_text(json.dumps(results, indent=0))
print("m3 records", len(results))

# ---- measurement 4, exhaustive at whole-variable scope (nothing bound)
spans = defaultdict(list)  # sv -> [(facet, years set)]
fy = t.group_by(["variable", "facet_id"]).aggregate(
    [("year_mask_0", "list"), ("year_mask_1", "list")]
)
for sv, fid, m0s, m1s in zip(
    fy["variable"].to_pylist(),
    fy["facet_id"].to_pylist(),
    fy["year_mask_0_list"].to_pylist(),
    fy["year_mask_1_list"].to_pylist(),
    strict=True,
):
    a = np.bitwise_or.reduce(np.array(m0s, dtype=np.uint64))
    b = np.bitwise_or.reduce(np.array(m1s, dtype=np.uint64))
    spans[sv].append((fid, frozenset(years_of(int(a), int(b)))))

groups = []
coverage_vars = set(spans)
for ds, fam, members in families(ci, coverage_vars):
    groups.append(("family", ds, members))
    rows_ = [ci.statvar_by_dcid(m) for m in members]
    for prop in sorted(fam.properties):
        g = defaultdict(list)
        for r in rows_:
            if prop in r.constraint_properties:
                g[
                    tuple(sorted((k, v) for k, v in r.constraint_properties.items() if k != prop))
                ].append(r.dcid)
        groups += [("siblings", ds, svs) for svs in g.values() if len(svs) >= 2]


def most(per_member, key):
    holders = defaultdict(set)
    for i, facets in enumerate(per_member):
        for fid, _ in facets:
            holders[key(fid)].add(i)
    if not holders:
        return frozenset()
    top = max(map(len, holders.values()))
    return frozenset(k for k, v in holders.items() if len(v) == top)


keys = {
    "index_key": lambda f: (meta[f].provenance_id, meta[f].unit) if f in meta else None,
    "facet_id_key": lambda f: f,
}
summary = Counter()
examples = defaultdict(list)
for kind, ds, members in groups:
    for wname, first, last in RANGES[:2]:
        life = [spans[sv] for sv in members]
        inside = [[(f, ys) for f, ys in fs if any(first <= y <= last for y in ys)] for fs in life]
        if not any(inside):
            continue
        summary[(kind, wname, "evaluated")] += 1
        for kn, key in keys.items():
            a, b = most(life, key), most(inside, key)
            if a != b:
                summary[(kind, wname, kn, "differ")] += 1
                if not (a & b):
                    summary[(kind, wname, kn, "disjoint")] += 1
                if len(examples[(kind, kn)]) < 5:
                    examples[(kind, kn)].append(
                        {
                            "window": wname,
                            "dataset": ds,
                            "members": members[:6],
                            "n": len(members),
                            "lifetime": sorted(map(str, a)),
                            "window_keys": sorted(map(str, b)),
                        }
                    )
(HERE / "m4_exhaustive.json").write_text(
    json.dumps(
        {
            "summary": {"|".join(k): v for k, v in summary.items()},
            "groups": len(groups),
            "examples": {"|".join(k): v for k, v in examples.items()},
        },
        indent=1,
    )
)
print(json.dumps({"|".join(k): v for k, v in summary.items()}, indent=0))
