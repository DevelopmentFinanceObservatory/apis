"""Measurements 2 and 4 over populations.json, three windows each.

2: the pair-plan election (`_member_groups`), the bulk election (`role_election`) and the
   mint's election with its member-count dispatch, compared on lead, per-member selected
   facet ids, unserved members and continuation.
4: for groups of two or more members, the agreement key the group elects when most-holders
   count over lifetime facets (today's `_group_pool`) against counting only facets with a
   point in the window whenever any member has one; under the index-sourced key
   (provenance, unit), the key these groups use, and under the facet-id key a recall-sourced
   group of the same members would use.

Writes m24_results.jsonl, one line per population and window.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace

from me_common import (
    WINDOWS,
    Population,
    Window,
    bulk_election,
    compare,
    confirmed_members,
    mint_election,
    pair_election,
    probes_for,
)
from me_load import HERE, load

from materialise.facets import overlaps_window
from materialise.group_facets import _agreement_key, _facet_id_key, elect_probes

only = sys.argv[1] if len(sys.argv) > 1 else None
if len(sys.argv) > 2:  # an extra window, e.g. 2022-2027, run alone
    a, b = map(int, sys.argv[2].split("-"))
    WINDOWS = (Window(sys.argv[2], a, b),)
loaded, views = load()
cov, ci = views.coverage, views.code_index
pops = json.loads((HERE / "populations.json").read_text())
if only:
    pops = [p for p in pops if p["source"] == only]
suffix = ("_" + only if only else "") + ("_" + sys.argv[2] if len(sys.argv) > 2 else "")
out = (HERE / f"m24_results{suffix}.jsonl").open("w")


def key_set(probes, key, window):
    """The most-held agreement keys over each member's facets, lifetime and window-narrowed."""

    def most(per_member):
        holders = {}
        for i, cands in enumerate(per_member):
            for c in cands:
                if c.facet_id is not None:
                    holders.setdefault(key(c), set()).add(i)
        if not holders:
            return frozenset(), 0
        top = max(len(v) for v in holders.values())
        return frozenset(k for k, v in holders.items() if len(v) == top), top

    lifetime = [p.candidates for p in probes.values()]
    inside = [[c for c in cands if overlaps_window(c, window=window)] for cands in lifetime]
    narrowed = inside if any(inside) else lifetime
    return most(lifetime), most(narrowed), narrowed


def lead_key(probes, *, index_sourced, window, candidates=None):
    if candidates is not None:
        probes = {
            sv: replace(p, candidates=tuple(c))
            for (sv, p), c in zip(probes.items(), candidates, strict=True)
        }
    g = elect_probes(probes, window=window, index_sourced=index_sourced)
    if g is None:
        return None
    sel = g.election.selections.get("provenance")
    return sel.dispatch_facet_id if sel else None


for n, raw in enumerate(pops):
    pop = Population(
        raw["name"],
        raw["members"],
        {k: frozenset(v) for k, v in raw["bound"].items()},
        raw["dataset"],
        raw["source"],
    )
    for w in WINDOWS:
        rec = {
            "population": pop.name,
            "source": pop.source,
            "window": w.name,
            "bound": raw["bound"],
        }
        try:
            members, rows = confirmed_members(cov, pop, w)
            rec["confirmed"] = len(members)
            if not members:
                out.write(json.dumps(rec) + "\n")
                continue
            probes, narrowed = probes_for(cov, ci, pop, members)
            pair = pair_election(narrowed, pop, w, members)
            bulk = bulk_election(cov, ci, pop, w, members, rows)
            try:
                mint = mint_election(probes, pop, w, members, ci)
            except Exception as exc:  # the mint refuses
                mint = None
                rec["mint_refused"] = f"{type(exc).__name__}: {str(exc)[:200]}"

            def enc(e):
                return (
                    None
                    if e is None
                    else {
                        "lead": e["lead"],
                        "selected": {k: sorted(v) for k, v in e["selected"].items()},
                        "unserved": sorted(e["unserved"]),
                        "continuing": sorted(e["continuing"]),
                    }
                )

            rec["pair"], rec["bulk"], rec["mint"] = enc(pair), enc(bulk), enc(mint)
            if mint is not None:
                rec["pair_vs_mint"] = compare(pair, mint)
                rec["bulk_vs_mint"] = compare(bulk, mint)
            rec["pair_vs_bulk"] = compare(pair, bulk)
            # facets each member's probe holds, for examples
            rec["facets"] = {
                sv: [
                    (c.facet_id, c.measurement_method, c.earliest_date, c.latest_date, c.obs_count)
                    for c in narrowed[sv].candidates
                ]
                for sv in members[:5]
            }
            if len(members) > 1:
                m4 = {}
                for kname, key, idx in (
                    ("index_key", _agreement_key, True),
                    ("facet_id_key", _facet_id_key, False),
                ):
                    (life, life_n), (win, win_n), cands = key_set(narrowed, key, w.requested)
                    life_lead = lead_key(narrowed, index_sourced=idx, window=w.requested)
                    win_lead = lead_key(
                        narrowed, index_sourced=idx, window=w.requested, candidates=cands
                    )
                    m4[kname] = {
                        "lifetime_keys": sorted(map(str, life)),
                        "lifetime_holders": life_n,
                        "window_keys": sorted(map(str, win)),
                        "window_holders": win_n,
                        "keys_differ": life != win,
                        "lifetime_lead": life_lead,
                        "window_lead": win_lead,
                        "lead_differs": life_lead != win_lead,
                    }
                rec["m4"] = m4
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["trace"] = traceback.format_exc()[-1500:]
        out.write(json.dumps(rec) + "\n")
    if n % 50 == 0:
        print(n, len(pops), flush=True)
        out.flush()
out.close()
print("done")
