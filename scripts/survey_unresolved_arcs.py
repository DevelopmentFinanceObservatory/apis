"""Survey unresolved arcs on StatisticalVariables against the live graph.

The reverse probe may miss anonymous, dcid-less targets. The seed survey
covers explicit corpus-case and fixture StatVars, so its result is a sample.
Both probes distinguish absent arcs from arcs whose nodes lack dcids.
The script throttles batches because LiveGraphClient leaves rate control to callers.

Usage:
    uv run --env-file .env python scripts/survey_unresolved_arcs.py
    uv run --env-file .env python scripts/survey_unresolved_arcs.py --skip-reverse
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from search.engine.clients.graph_client import LiveGraphClient
from search.eval.cases import CandidatesExpect, load_cases
from search.eval.live_clients import bind_run_catalogue

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_NODES_PATH = _REPO_ROOT / "tests" / "fixtures" / "graph_nodes.json"

_THROTTLE_S = 0.3
_BATCH_SIZE = 40
# Reverse enumeration covers more than 20,000 instances, with hundreds per page.
# The 1,000-page cap leaves room above that count while bounding malformed runs.
_DEFAULT_MAX_REVERSE_PAGES = 1_000
# A typical run uses about 1,040 requests: 40 enumeration pages plus two reads
# for roughly 500 batches of 40. The 1,200 cap leaves room for growth and stops
# unexpectedly broad or looping runs.
_DEFAULT_MAX_REVERSE_REQUESTS = 1_200

ArcState = Literal["resolved", "unresolved", "absent"]


def classify_arc(arcs: dict, prop: str) -> ArcState:
    """Return `resolved`, `unresolved`, or `absent` for one arc.

    An unresolved arc contains dcid-less nodes. An absent arc contains zero nodes.
    Keeping both states prevents an asserted unresolved denominator from
    appearing missing.
    """
    nodes = arcs.get(prop, {}).get("nodes", [])
    if not nodes:
        return "absent"
    if any("dcid" in n for n in nodes):
        return "resolved"
    return "unresolved"


def seed_sv_dcids() -> list[str]:
    """Return the StatVars the corpus cases name, unioned with fixture StatisticalVariables."""
    seeds: set[str] = set()

    for case in load_cases():
        for scope in case.expect.parts or [case.expect]:
            for status in scope.accepted_statuses:
                block = scope.for_status(status)
                if block is None:
                    continue
                forms = list(block.measure.forms) if block.measure is not None else []
                if isinstance(block, CandidatesExpect):
                    forms.extend(block.family_forms.values())
                for form in forms:
                    seeds.update(sv for sv in form.stat_vars or [] if isinstance(sv, str))

    fixture_nodes = json.loads(_FIXTURE_NODES_PATH.read_text())
    for dcid, node in fixture_nodes.items():
        if isinstance(node, dict) and node.get("type") == "StatisticalVariable":
            seeds.add(dcid)

    return sorted(seeds)


class SurveyResult(NamedTuple):
    """Per-property state counts, unresolved pairs in survey order, and classified SV count.

    `n_classified` counts SVs with batch payloads.
    `unresolved_pairs` retains each affected SV for follow-up.
    """

    counts: dict[str, Counter]
    unresolved_pairs: list[tuple[str, str]]
    n_classified: int


def survey(graph: LiveGraphClient, sv_dcids: list[str]) -> SurveyResult:
    """Batch-fetch and classify denominator, qualifier, and constraint-property arcs.

    Sleeps `_THROTTLE_S` between `_BATCH_SIZE`-sized batches.
    """
    counts: dict[str, Counter] = {}
    unresolved_pairs: list[tuple[str, str]] = []
    n_classified = 0
    for i in range(0, len(sv_dcids), _BATCH_SIZE):
        batch = sv_dcids[i : i + _BATCH_SIZE]
        arcs_by_dcid = graph.node_arcs_batch(batch)
        for sv_dcid, arcs in arcs_by_dcid.items():
            if arcs is None:
                continue
            n_classified += 1
            constraint_props = [
                n["dcid"]
                for n in arcs.get("constraintProperties", {}).get("nodes", [])
                if "dcid" in n
            ]
            for prop in ("measurementDenominator", "measurementQualifier", *constraint_props):
                state = classify_arc(arcs, prop)
                counts.setdefault(prop, Counter())[state] += 1
                if state == "unresolved":
                    unresolved_pairs.append((sv_dcid, prop))
        if i + _BATCH_SIZE < len(sv_dcids):
            time.sleep(_THROTTLE_S)
    return SurveyResult(counts=counts, unresolved_pairs=unresolved_pairs, n_classified=n_classified)


class ReversePaginationError(RuntimeError):
    """A paginated reverse query cannot be reported as complete."""


@dataclass
class _ReverseRequestBudget:
    """Run-scoped cap on raw reverse /v2/node requests."""

    limit: int
    remaining: int

    @classmethod
    def create(cls, limit: int) -> _ReverseRequestBudget:
        if limit < 1:
            raise ValueError("max_requests must be positive")
        return cls(limit=limit, remaining=limit)

    def consume(self, description: str) -> None:
        self.ensure_available(description)
        self.remaining -= 1

    def ensure_available(self, description: str) -> None:
        if self.remaining < 1:
            raise ReversePaginationError(
                f"reverse request budget of {self.limit} exhausted before {description}; "
                "refusing to report a partial result"
            )


def _reverse_node_pages(
    graph: LiveGraphClient,
    *,
    nodes: list[str],
    prop: str,
    max_pages: int,
    description: str,
    request_budget: _ReverseRequestBudget,
):
    """Yield pages for one reverse `/v2/node` request.

    Each continuation resends the same nodes and property, changing only
    `nextToken`. `max_pages` bounds each enumeration or batch request.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be positive")

    seen_tokens: set[str] = set()
    next_token: str | None = None
    for page_number in range(1, max_pages + 1):
        payload: dict[str, object] = {"nodes": nodes, "property": prop}
        if next_token is not None:
            payload["nextToken"] = next_token

        request_budget.consume(description)
        body = graph._client.post("/v2/node", payload)
        yield body

        raw_next_token = body.get("nextToken")
        if raw_next_token is None or raw_next_token == "":
            return
        if not isinstance(raw_next_token, str):
            raise ReversePaginationError(f"{description} returned a non-string top-level nextToken")
        if raw_next_token in seen_tokens:
            raise ReversePaginationError(
                f"{description} repeated a nextToken; refusing to report a partial population"
            )
        if page_number == max_pages:
            raise ReversePaginationError(
                f"{description} reached the {max_pages}-page limit; rerun with a higher "
                "--max-pages value rather than treating this partial result as exhaustive"
            )

        seen_tokens.add(raw_next_token)
        next_token = raw_next_token
        request_budget.ensure_available(description)
        time.sleep(_THROTTLE_S)


def _reverse_provisional_instance_dcids(
    graph: LiveGraphClient,
    *,
    max_pages: int,
    request_budget: _ReverseRequestBudget,
) -> list[str]:
    """Enumerate reachable ProvisionalNode ids in first-seen order.

    Reads the raw response through `graph._client.post` to follow top-level
    `nextToken` pagination. Raises `ReversePaginationError` for malformed or
    repeated tokens and an exhausted page bound.
    """
    instance_dcids: list[str] = []
    seen_dcids: set[str] = set()
    for body in _reverse_node_pages(
        graph,
        nodes=["ProvisionalNode"],
        prop="<-typeOf",
        max_pages=max_pages,
        description="reverse ProvisionalNode enumeration",
        request_budget=request_budget,
    ):
        arcs = body.get("data", {}).get("ProvisionalNode", {}).get("arcs", {})
        nodes = arcs.get("typeOf", {}).get("nodes", [])
        for node in nodes:
            dcid = node.get("dcid")
            if isinstance(dcid, str) and dcid not in seen_dcids:
                seen_dcids.add(dcid)
                instance_dcids.append(dcid)

    return instance_dcids


def reverse_probe(
    graph: LiveGraphClient,
    *,
    max_pages: int,
    max_requests: int = _DEFAULT_MAX_REVERSE_REQUESTS,
) -> tuple[list[str], int]:
    """Probe reverse denominator and qualifier arcs for ProvisionalNode ids.

    Returns the ids and hit count. An empty result is valid.
    `max_pages` bounds each request and `max_requests` bounds the whole probe.
    """
    request_budget = _ReverseRequestBudget.create(max_requests)
    instance_dcids = _reverse_provisional_instance_dcids(
        graph,
        max_pages=max_pages,
        request_budget=request_budget,
    )
    n_hits = 0
    for i in range(0, len(instance_dcids), _BATCH_SIZE):
        batch = instance_dcids[i : i + _BATCH_SIZE]
        for prop in ("<-measurementDenominator", "<-measurementQualifier"):
            for body in _reverse_node_pages(
                graph,
                nodes=batch,
                prop=prop,
                max_pages=max_pages,
                description=f"reverse {prop} batch probe",
                request_budget=request_budget,
            ):
                data = body.get("data", {})
                for dcid in batch:
                    node_data = data.get(dcid)
                    if node_data is None:
                        continue
                    arcs = node_data.get("arcs", {})
                    if arcs:
                        n_hits += sum(len(entry.get("nodes", [])) for entry in arcs.values())
        if i + _BATCH_SIZE < len(instance_dcids):
            request_budget.ensure_available("next reverse-property batch probe")
            time.sleep(_THROTTLE_S)
    return instance_dcids, n_hits


def _positive_page_count(value: str) -> int:
    """argparse validator for a finite, positive pagination bound."""
    try:
        page_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if page_count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return page_count


def _positive_request_count(value: str) -> int:
    """argparse validator for a finite, positive reverse request bound."""
    try:
        request_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if request_count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return request_count


def format_report(result: SurveyResult) -> str:
    lines = [f"=== unresolved-arc survey ({result.n_classified} SVs classified) ==="]
    header = f"{'property':<32}{'resolved':>10}{'unresolved':>12}{'absent':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for prop in sorted(result.counts):
        c = result.counts[prop]
        lines.append(f"{prop:<32}{c['resolved']:>10}{c['unresolved']:>12}{c['absent']:>10}")
    if result.unresolved_pairs:
        lines.append("")
        lines.append(f"unresolved (sv_dcid, property) pairs ({len(result.unresolved_pairs)}):")
        for sv_dcid, prop in result.unresolved_pairs:
            lines.append(f"  {sv_dcid}  {prop}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-reverse",
        action="store_true",
        help="Skip strategy 1 (the reverse <-typeOf probe) and run only the seed survey.",
    )
    parser.add_argument(
        "--max-pages",
        type=_positive_page_count,
        default=_DEFAULT_MAX_REVERSE_PAGES,
        help=(
            "Maximum /v2/node pages for each reverse enumeration or reverse-property batch "
            "probe request "
            f"(default: {_DEFAULT_MAX_REVERSE_PAGES}). The script stops with an error, "
            "rather than reporting a partial enumeration as exhaustive, if this is reached."
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=_positive_request_count,
        default=_DEFAULT_MAX_REVERSE_REQUESTS,
        help=(
            "Total raw /v2/node request budget shared by the full reverse probe "
            f"(default: {_DEFAULT_MAX_REVERSE_REQUESTS}). The script stops before an "
            "additional request would exceed it."
        ),
    )
    args = parser.parse_args()

    bind_run_catalogue()
    graph = LiveGraphClient.from_search_settings()
    try:
        if not args.skip_reverse:
            instance_dcids, n_hits = reverse_probe(
                graph,
                max_pages=args.max_pages,
                max_requests=args.max_requests,
            )
            print(
                f"reverse strategy: {len(instance_dcids)} ProvisionalNode instance(s) "
                f"found via <-typeOf, {n_hits} reverse measurementDenominator/"
                "measurementQualifier hit(s)"
            )
            if n_hits == 0:
                print(
                    "reverse strategy found no usable reverse arcs (expected -- per-SV "
                    "provisional targets are typically anonymous, dcid-less nodes "
                    "unreachable by a reverse query); falling back to the seed strategy"
                )
            print()

        sv_dcids = seed_sv_dcids()
        result = survey(graph, sv_dcids)
        print(format_report(result))
    finally:
        graph.close()


if __name__ == "__main__":
    main()
