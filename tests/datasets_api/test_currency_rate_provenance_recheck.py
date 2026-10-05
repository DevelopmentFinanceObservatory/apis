"""Direct tests for `recheck_rate_provenance`, including batched graph reads and reuse of
provenance dates already returned by the facet probe.
"""

from __future__ import annotations

from collections.abc import Sequence

from datasets_api.currency import recheck_rate_provenance
from datasets_api.versions import MemberSignal
from materialise.graph import ProvenanceRefresh

_FRESH = (MemberSignal("facet1", 10, "2020", "2021"),)


class _CountingGraph:
    def __init__(self, arcs: dict[str, dict[str, tuple[str, ...]]]) -> None:
        self._arcs = arcs
        self.calls: list[Sequence[str]] = []

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self.calls.append(tuple(dcids))
        result = {}
        for dcid in dcids:
            arcs = self._arcs.get(dcid)
            result[dcid] = (
                None if arcs is None else {prop: arcs.get(prop, ()) for prop in properties}
            )
        return result

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        return {
            dcid: tuple(
                dict.fromkeys(
                    value
                    for prop in ("name", "alternateName")
                    for value in self._arcs.get(dcid, {}).get(prop, ())
                )
            )
            for dcid in dict.fromkeys(dcids)
        }


def test_returns_signals_unchanged_when_the_stale_signal_carries_no_rate_provenance():
    stale = (MemberSignal("facet1", 10, "2020", "2021"),)
    graph = _CountingGraph({})

    result = recheck_rate_provenance(_FRESH, stale, graph=graph, provenance_refresh={})

    assert result == _FRESH
    assert graph.calls == []


def test_reads_a_rate_provenance_not_covered_by_the_facet_probe_in_one_extra_call():
    stale = (
        MemberSignal(
            "facet1", 10, "2020", "2021", rate_provenance_refresh=(("RATE_A", "2026-01-01"),)
        ),
    )
    graph = _CountingGraph({"RATE_A": {"lastDataRefreshDate": ("2026-02-01",)}})

    result = recheck_rate_provenance(_FRESH, stale, graph=graph, provenance_refresh={})

    assert result[0].rate_provenance_refresh == (("RATE_A", "2026-02-01"),)
    assert graph.calls == [("RATE_A",)]


def test_reuses_the_facet_probes_own_provenance_refresh_without_a_graph_call():
    stale = (
        MemberSignal(
            "facet1", 10, "2020", "2021", rate_provenance_refresh=(("RATE_A", "2026-01-01"),)
        ),
    )
    graph = _CountingGraph({})
    probe_refresh = {"RATE_A": ProvenanceRefresh("2026-01-01", None)}

    result = recheck_rate_provenance(_FRESH, stale, graph=graph, provenance_refresh=probe_refresh)

    assert result[0].rate_provenance_refresh == (("RATE_A", "2026-01-01"),)
    assert graph.calls == []


def test_batches_every_members_own_missing_ids_into_one_call():
    stale = (
        MemberSignal(
            "facet1", 10, "2020", "2021", rate_provenance_refresh=(("RATE_A", "2026-01-01"),)
        ),
        MemberSignal(
            "facet2", 5, "2020", "2021", rate_provenance_refresh=(("RATE_B", "2026-01-01"),)
        ),
    )
    fresh = (
        MemberSignal("facet1", 10, "2020", "2021"),
        MemberSignal("facet2", 5, "2020", "2021"),
    )
    graph = _CountingGraph(
        {
            "RATE_A": {"lastDataRefreshDate": ("2026-01-01",)},
            "RATE_B": {"lastDataRefreshDate": ("2026-02-01",)},
        }
    )

    result = recheck_rate_provenance(fresh, stale, graph=graph, provenance_refresh={})

    assert result[0].rate_provenance_refresh == (("RATE_A", "2026-01-01"),)
    assert result[1].rate_provenance_refresh == (("RATE_B", "2026-02-01"),)
    assert graph.calls == [("RATE_A", "RATE_B")]


def test_a_provenance_the_graph_no_longer_holds_recheck_reads_none():
    stale = (
        MemberSignal(
            "facet1", 10, "2020", "2021", rate_provenance_refresh=(("RATE_A", "2026-01-01"),)
        ),
    )
    graph = _CountingGraph({})  # "RATE_A" carries no entry at all.

    result = recheck_rate_provenance(_FRESH, stale, graph=graph, provenance_refresh={})

    assert result[0].rate_provenance_refresh == (("RATE_A", None),)


def test_a_length_mismatch_against_the_stale_signal_returns_signals_unchanged():
    stale = (
        MemberSignal(
            "facet1", 10, "2020", "2021", rate_provenance_refresh=(("RATE_A", "2026-01-01"),)
        ),
        MemberSignal("facet2", 5, "2020", "2021"),
    )
    graph = _CountingGraph({})

    result = recheck_rate_provenance(_FRESH, stale, graph=graph, provenance_refresh={})

    assert result == _FRESH
    assert graph.calls == []
