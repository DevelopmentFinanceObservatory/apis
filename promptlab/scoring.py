"""Scores that read stored LLM calls rather than a resolve response.

`seed_flip_rate` measures how often the same case's seeds disagree. `make_nomination_correctness`
scores the bind stage's `also_plausible` nominations against the menu the model saw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langfuse import Evaluation

from promptlab.bindmenu import MenuEntry
from promptlab.expectations import LegExpectation
from search.engine.clients.graph import EngineGraphClient
from search.engine.retrieval.facts import read_five_tuple
from search.eval.runner import Sample, modal_sample


def seed_flip_rate(outputs: Sequence[Mapping]) -> float | None:
    """Return `1 - modal_count / run_count` for one case's seed outputs.

    Outputs group by the runner's answer key, the vote that picks a case's modal sample. A
    single run gives no variance estimate, so it returns None.
    """
    if len(outputs) <= 1:
        return None
    samples = [
        Sample(
            body=dict(output),
            status=200,
            degraded=False,
            request_rejected=False,
            attempts=1,
            elapsed_s=0.0,
        )
        for output in outputs
    ]
    _index, n_agreeing = modal_sample(samples)
    return 1 - n_agreeing / len(outputs)


# How the bind menu shows a denominator arc whose target is unresolved, whatever its dcid.
UNRESOLVED_DENOMINATOR = "(present, unresolved)"


def menu_denom_display(dcid: str | None, unresolved: bool) -> str | None:
    """Match bind-menu encoding: unresolved sentinel, resolved dcid, or None for absent."""
    if unresolved:
        return UNRESOLVED_DENOMINATOR
    if dcid:
        return dcid
    return None


def menu_five_tuple(
    entry: MenuEntry,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Return the five tuple fields, normalising empty values to None."""
    ft = entry.five_tuple
    return (
        ft.get("population_type") or None,
        ft.get("measured_property") or None,
        ft.get("stat_type") or None,
        ft.get("measurement_qualifier") or None,
        ft.get("measurement_denominator") or None,
    )


def _resolve_menu_index(
    dcid: str, menu: Sequence[MenuEntry], graph: EngineGraphClient
) -> int | None:
    """Match a StatVar's graph five-tuple to a menu index, or return None.

    Falls back to ``sample_dcids``, which the bind prompt caps at five and may truncate.
    Graph errors propagate to the caller.
    """
    arcs = graph.node_arcs(dcid)
    if arcs is not None:
        target = read_five_tuple(arcs)
        target_tuple = (
            target.pop_type_dcid or None,
            target.meas_prop_dcid or None,
            target.stat_type_dcid or None,
            target.meas_qual_dcid,
            menu_denom_display(target.meas_denom_dcid, target.meas_denom_unresolved),
        )
        for entry in menu:
            if menu_five_tuple(entry) == target_tuple:
                return entry.index
    for entry in menu:
        if dcid in entry.sample_dcids:
            return entry.index
    return None


def make_nomination_correctness(graph: EngineGraphClient):
    """Score bind-stage nominations against the menu and a leg's expected StatVars.

    A leg that accepts only `candidates` gets recall over its StatVars that resolve to menu
    entries; unresolved ids are reported separately, and if none resolve the result is
    categorical ``unscorable``. A leg that accepts only `definite` gets the raw nomination
    count; other legs are unscored.
    """

    def nomination_correctness(
        *, output: Mapping, expected: LegExpectation, menu: Sequence[MenuEntry]
    ) -> list[Evaluation]:
        also_plausible: list[int] = output.get("also_plausible") or []

        if expected.sole_status == "definite":
            return [Evaluation(name="over_nomination_count", value=len(also_plausible))]

        if expected.sole_status != "candidates":
            return []

        resolved_indices: list[int] = []
        unscorable: list[str] = []
        for dcid in expected.stat_vars:
            idx = _resolve_menu_index(dcid, menu, graph)
            if idx is None:
                unscorable.append(dcid)
            else:
                resolved_indices.append(idx)

        if not resolved_indices:
            return [
                Evaluation(
                    name="expected_alternate_recall",
                    value="unscorable",
                    data_type="CATEGORICAL",
                    comment=f"no expected alternate resolved to a menu index: {expected.stat_vars}",
                    metadata={"unscorable_dcids": unscorable},
                )
            ]

        hits = sum(1 for idx in resolved_indices if idx in also_plausible)
        comment = f"{hits}/{len(resolved_indices)} expected alternates found in also_plausible"
        if unscorable:
            comment += f"; unscorable={unscorable}"
        return [
            Evaluation(
                name="expected_alternate_recall",
                value=hits / len(resolved_indices),
                comment=comment,
                metadata={"resolved": len(resolved_indices), "unscorable_dcids": unscorable},
            )
        ]

    return nomination_correctness
