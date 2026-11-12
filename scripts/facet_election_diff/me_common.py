"""Populations, windows and the three elections (pair plan, bulk plan, mint) over the staged
release's coverage view. No declared defaults exist on the dev graph (me_declared.py read
none), so every election runs with no declared policy."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from materialise import MaterialisationRequest
from materialise.facets import (
    ElectedDataset,
    continuing_facet_ids,
    joint_dispatch_facet_ids,
)
from materialise.generation import _select_member_facets
from materialise.group_facets import group_breadth, group_variable_breadth, select_group_facets
from materialise.window import RequestedWindow
from search.engine.retrieval.elected_dataset import published_by_stat_var
from search.engine.retrieval.read_plan import _member_groups, plan_declared_defaults, role_election
from search.engine.retrieval.role_scope import (
    _role_index_from_coverage,
    bulk_candidates,
    member_probes,
)
from search.models import TimeWindow
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------- memory


class _RusageV2(ctypes.Structure):
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("pkg_idle_wkups", ctypes.c_uint64),
        ("interrupt_wkups", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("wired_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
        ("pad", ctypes.c_uint64 * 40),
    ]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def mem_now() -> tuple[int, int]:
    """(resident bytes, physical footprint bytes) of this process now."""
    info = _RusageV2()
    _libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info))
    return info.resident_size, info.phys_footprint


class PeakSampler:
    """Samples resident size and footprint every 5 ms while open; reports peak over the
    value at entry."""

    def __enter__(self):
        self.base = mem_now()
        self.peak = list(self.base)
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        self.t0 = time.perf_counter()
        return self

    def _run(self):
        while not self._stop:
            r, f = mem_now()
            self.peak[0] = max(self.peak[0], r)
            self.peak[1] = max(self.peak[1], f)
            time.sleep(0.005)

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self._stop = True
        self._t.join()
        r, f = mem_now()
        self.peak = [max(self.peak[0], r), max(self.peak[1], f)]
        self.end = (r, f)
        return False

    def report(self) -> dict:
        mib = 2**20
        return {
            "seconds": round(self.seconds, 3),
            "base_rss_mib": round(self.base[0] / mib),
            "base_footprint_mib": round(self.base[1] / mib),
            "peak_rss_delta_mib": round((self.peak[0] - self.base[0]) / mib, 1),
            "peak_footprint_delta_mib": round((self.peak[1] - self.base[1]) / mib, 1),
            "end_footprint_delta_mib": round((self.end[1] - self.base[1]) / mib, 1),
        }


# ---------------------------------------------------------------- windows


@dataclass(frozen=True)
class Window:
    name: str
    first: int | None  # None for latest
    last: int | None

    @property
    def plan_years(self):
        return (
            "latest"
            if self.first is None
            else TimeWindow(start_year=self.first, end_year=self.last)
        )

    @property
    def read_years(self):
        return (
            None if self.first is None else tuple(str(y) for y in range(self.first, self.last + 1))
        )

    @property
    def requested(self) -> RequestedWindow:
        if self.first is None:
            return RequestedWindow(kind="latest")
        return RequestedWindow(kind="range", start_year=self.first, end_year=self.last)


WINDOWS = (
    Window("latest", None, None),
    Window("2019-2023", 2019, 2023),
    Window("1995-2000", 1995, 2000),
)


# ---------------------------------------------------------------- populations


@dataclass
class Population:
    name: str
    members: list[str]
    bound: dict = field(default_factory=dict)  # coverage roles -> frozenset
    dataset: str | None = None  # elected dataset provenance; None = not index-sourced
    source: str = ""


def families(ci, coverage_vars: set[str]):
    """(dataset, family, members in coverage) for every index family and dataset."""
    for fams in ci.families_by_five_tuple.values():
        for fam in fams:
            for ds in fam.datasets:
                members = [
                    r.dcid for r in fam.rows if ds in r.provenances and r.dcid in coverage_vars
                ]
                if members:
                    yield ds, fam, members


def elected(pop: Population, ci) -> dict[str, ElectedDataset]:
    if pop.dataset is None:
        return {}
    out = {}
    for sv in pop.members:
        row = ci.statvar_by_dcid(sv)
        out[sv] = ElectedDataset(
            provenance=pop.dataset, member_provenances=row.provenances if row else (pop.dataset,)
        )
    return out


# ---------------------------------------------------------------- elections


def confirmed_members(cov, pop: Population, w: Window) -> list[str]:
    """Members holding a row in the window under the bound roles (the plan's confirmed SVs)."""
    rows = cov.pair_rows(
        variables=pop.members, bound=pop.bound, first_year=w.first, last_year=w.last
    )
    held = {g.variable for g in rows.groups()}
    return [sv for sv in pop.members if sv in held], rows


def _role_names(cov, members):
    rn = cov.role_names[members[0]]
    return rn


def bulk_election(cov, ci, pop: Population, w: Window, members, rows):
    rn = _role_names(cov, members)
    # The open axis: the one not bound, among the roles the variable declares.
    member_is_donor = rn.recipient is None or ("recipient" in pop.bound and rn.donor is not None)
    role_name = rn.donor if member_is_donor else rn.recipient
    opposite = rn.recipient if member_is_donor else rn.donor
    datasets = elected(pop, ci)
    index = published_by_stat_var(
        datasets,
        _role_index_from_coverage(
            members,
            rows,
            facet_metadata=cov.facet_metadata,
            role_name=role_name,
            opposite_role_name=opposite,
            member_is_donor=member_is_donor,
        ),
    )
    confirmed = [sv for sv in members if index[sv]]
    by_sv = {sv: index[sv] for sv in confirmed}
    cands = bulk_candidates(by_sv, confirmed, bound=pop.bound, coverage_index=cov, declared={})
    e = role_election(
        cands,
        members=confirmed,
        read_years=w.read_years,
        index_sourced=pop.dataset is not None,
        declared={},
    )
    return {
        "lead": e.facet_id,
        "selected": {sv: frozenset(ids) for sv, ids in e.selected.items()},
        "unserved": frozenset(e.unserved),
        "continuing": frozenset(e.continuing),
    }


def probes_for(cov, ci, pop: Population, members):
    probes = member_probes(cov, variables=members, bound=pop.bound)
    datasets = elected(pop, ci)
    narrowed = {
        sv: replace(
            p,
            candidates=tuple(
                c
                for c in p.candidates
                if sv not in datasets or datasets[sv].publishes(c.provenance_id)
            ),
        )
        for sv, p in probes.items()
    }
    return probes, narrowed


def pair_election(narrowed, pop: Population, w: Window, members):
    """The pair plan's election through its own seam (`read_plan._elect_groups`, which decides
    the one-member dispatch), with the lead and continuation read from the same group."""
    import inspect

    from materialise.group_facets import elect_probes
    from search.engine.retrieval import read_plan
    from search.engine.retrieval.read_plan import _elect_groups, _pair_populated, plan_window

    index_sourced = pop.dataset is not None
    defaults = plan_declared_defaults({}, _pair_populated((), narrowed))
    _series, _scope, selections, unserved = _elect_groups(
        (),
        members=members,
        years=w.plan_years,
        scope=False,
        probes=narrowed,
        index_sourced=index_sourced,
        declared_defaults=defaults,
    )
    # The flag `_elect_groups` hands `_member_groups` on this head.
    effective = index_sourced and (
        len(members) > 1
        if "len(members) > 1" in inspect.getsource(read_plan._elect_groups)
        else True
    )
    groups, check, _ = _member_groups(
        narrowed, years=w.plan_years, index_sourced=effective, declared_defaults=defaults
    )
    assert check == selections, "harness flag diverges from _elect_groups"
    g = elect_probes(
        narrowed, window=plan_window(w.plan_years), index_sourced=effective, policies=defaults
    )
    continuing = frozenset(i for ids in (g.continuing_ids if g else ()) for i in ids)
    lead = next((grp[0].facet_id for grp in groups.values() if grp), None)
    return {
        "lead": lead,
        "selected": {sv: frozenset(s) for sv, s in selections.items()},
        "unserved": frozenset(unserved),
        "continuing": continuing,
    }


def _request(sv, pop: Population, w: Window, datasets):
    ds = datasets.get(sv)
    return MaterialisationRequest(
        stat_var=sv,
        role_bindings={},
        declared_roles=("financingProvider", "financingRecipient"),
        facet_bindings={},
        window=w.requested,
        index_sourced=pop.dataset is not None,
        dataset=ds,
    )


def mint_election(probes, pop: Population, w: Window, members, ci):
    graph = FakeGraphReader()
    datasets = elected(pop, ci)
    reqs = [_request(sv, pop, w, datasets) for sv in members]
    plist = [probes[sv] for sv in members]
    if len(members) == 1:
        from materialise.facets import dataset_candidates

        p = plist[0]
        el = _select_member_facets(
            reqs[0],
            p.candidates,
            graph=graph,
            breadth=p.breadth_in(w.requested),
            variable_breadth=p.variable_breadth,
        )
        cands = dataset_candidates(reqs[0], p.candidates)
        disp = joint_dispatch_facet_ids(cands, el, member=0)
        cont = continuing_facet_ids(cands, el, member=0)
        sel = frozenset(disp) - frozenset(cont)
        return {
            "lead": el.selections["provenance"].dispatch_facet_id,
            "selected": {members[0]: sel} if sel else {},
            "unserved": frozenset() if disp else frozenset(members),
            "continuing": frozenset(cont),
            "method": el.selections.get("measurementMethod").ref
            if el.selections.get("measurementMethod")
            else None,
        }
    group = select_group_facets(
        reqs,
        [p.candidates for p in plist],
        graph=graph,
        breadth=group_breadth(plist, window=w.requested),
        variable_breadth=group_variable_breadth(plist),
    )
    sel = {sv: group.selected(i) for i, sv in enumerate(members) if group.selected(i)}
    prov = group.election.selections.get("provenance")
    return {
        "lead": prov.dispatch_facet_id if prov else None,
        "selected": sel,
        "unserved": frozenset(members[i] for i in group.unserved),
        "continuing": frozenset(i for ids in group.continuing_ids for i in ids),
    }


def compare(a: dict, b: dict) -> dict:
    """Disagreements between two elections over the same population and window."""
    members = set(a["selected"]) | set(b["selected"])
    differing = sorted(
        sv
        for sv in members
        if a["selected"].get(sv, frozenset()) != b["selected"].get(sv, frozenset())
    )
    return {
        "lead": a["lead"] != b["lead"],
        "members": differing,
        "unserved": a["unserved"] != b["unserved"],
        "continuation": a["continuing"] != b["continuing"],
    }


def corpus_populations(ci, coverage_vars, corpus_dir) -> list[Population]:
    """Each corpus case expecting an index family, as that family's minimal-constraint rows
    holding its fixed and bound values, under the case's named entities. An approximation of
    the answer's members: the resolver's role-class election is not rerun."""
    import glob

    import yaml

    fams = {}
    for f in glob.glob(f"{corpus_dir}/families/*.yaml"):
        fams.update(yaml.safe_load(Path(f).read_text()) or {})
    pops = []
    for f in sorted(glob.glob(f"{corpus_dir}/*.yaml")):
        for case in yaml.safe_load(Path(f).read_text()) or []:
            d = (case.get("expect") or {}).get("definite") or {}
            m = d.get("measure") or {}
            fl = m.get("family")
            if not fl:
                continue
            fl = fl if isinstance(fl, list) else [fl]
            bound_props = m.get("bound") or {}
            bound = {}
            for e in d.get("entities") or []:
                role = {"origin": "donor", "destination": "recipient"}.get(e.get("role"))
                if role and isinstance(e.get("dcid"), str):
                    bound.setdefault(role, set()).add(e["dcid"])
            for fam_name in fl[:1]:  # the family the case lists first
                spec = fams.get(fam_name)
                if (
                    not spec
                    or not isinstance(spec.get("dataset"), str)
                    or "shape" not in spec
                    or "gl" in (spec.get("fixed") or {})
                ):
                    continue
                shape = spec["shape"]
                need = {
                    **(spec.get("fixed") or {}),
                    **{k: v for k, v in bound_props.items() if isinstance(v, str)},
                }
                rows = [
                    r
                    for r in ci.statvars
                    if spec["dataset"] in r.provenances
                    and r.dcid in coverage_vars
                    and r.population_type == shape["population_type"]
                    and r.measured_property == shape["measured_property"]
                    and (r.measurement_denominator or None) == shape.get("measurement_denominator")
                    and all(r.constraint_properties.get(k) == v for k, v in need.items())
                ]
                if not rows:
                    continue
                fewest = min(len(r.constraint_properties) for r in rows)
                members = [r.dcid for r in rows if len(r.constraint_properties) == fewest]
                pops.append(
                    Population(
                        name=f"corpus:{case['id']}",
                        members=members,
                        bound={k: frozenset(v) for k, v in bound.items()},
                        dataset=spec["dataset"],
                        source="corpus",
                    )
                )
    return pops
