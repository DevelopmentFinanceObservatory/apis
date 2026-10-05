"""Send the sentence panel through the deployed Search resolver and digest the sentences.

The panel is a fixed set of queries in `scripts/sentence_panel/queries/*.yaml`, one file per
slice (`crs`, `mls`, `ids`, `weo`). Each query carries an id, the axes it exercises, the
intent behind it, and optionally the edge case it tests. `scripts/sentence_panel/rubric.md`
is the rubric for scoring a run.

For each slice the script posts every query to `/api/search/resolve` with `include_sentence`
and writes three files to `--out`:

- `raw-<slice>.jsonl`: the full response per query.
- `digest-<slice>.jsonl`: one structured row per query.
- `digest-<slice>.md`: one block per query with its status, sentences, what was bound, and
  what the response offers to narrow further.

It reads the live deployed service and takes about five minutes for the whole panel. It prints,
per slice, the query count, transport errors and status counts. It exits non-zero when a
queries file is empty or every query in a slice fails at transport level.

`--redigest` rebuilds the digests from the saved raw files in `--out` without calling the
resolver, which is how to pick up a change to the digest code or to a query's metadata.

Usage:
    just sentence-panel
    just sentence-panel crs ids
    just sentence-panel --redigest --out runs/sentence-panel

Equivalent without `just`:
    uv run python scripts/sentence_panel/run.py [slice ...] [--out DIR] [--base URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import yaml

QUERIES_DIR = Path(__file__).resolve().parent / "queries"
DEFAULT_OUT = Path("runs/sentence-panel")
DEFAULT_BASE = "https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app"
DEFAULT_CONCURRENCY = 4
TIMEOUT_S = 150
ATTEMPTS = 2


async def resolve(client: httpx.AsyncClient, base: str, query: str) -> dict[str, Any]:
    body = {
        "input": {"kind": "raw_text", "query": query},
        "response_options": {"include_sentence": True},
    }
    last_error: Exception | None = None
    for _ in range(ATTEMPTS):
        try:
            r = await client.post(f"{base}/api/search/resolve", json=body, timeout=TIMEOUT_S)
            return {"http_status": r.status_code, "body": r.json()}
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
    return {"http_status": None, "error": repr(last_error)}


def _label(ref: dict | None) -> str | None:
    if not ref:
        return None
    return ref.get("label") or ref.get("dcid")


def _binding_text(binding: dict) -> str:
    kind = binding.get("kind")
    if kind == "value":
        return _value_text(binding["value"])
    if kind == "set":
        vals = binding.get("values") or binding.get("value") or []
        return "{" + ", ".join(_value_text(v) for v in vals) + "}"
    return kind or "?"


def _value_text(value: dict) -> str:
    if value.get("ref"):
        return _label(value["ref"]) or "?"
    return json.dumps(value)


def _cap(items: list, n: int = 10) -> list:
    return items if len(items) <= n else [*items[:n], f"... and {len(items) - n} more"]


def digest_spec(spec: dict) -> dict:
    res = spec.get("resolved") or {}
    slots = res.get("slots") or []
    bound = {
        s["key"]["label"]: _binding_text(s["binding"])
        for s in slots
        if s["binding"].get("kind") != "unbound"
    }
    unbound = [s["key"]["label"] for s in slots if s["binding"].get("kind") == "unbound"]
    slot_options = {}
    for so in res.get("slot_options") or []:
        opts = so.get("options") or []
        slot_options[so["key"]["label"]] = {
            "n": len(opts),
            "sample": [_value_text(o["value"]) for o in opts[:6] if o.get("value")],
        }
    facets = res.get("facets") or {}
    return {
        "measure_label": res.get("measure_label"),
        "shape_label": (res.get("shape") or {}).get("label"),
        "sentence": res.get("rendered_sentence"),
        "entities": _cap(
            [f"{e.get('label')} ({e.get('role')})" for e in res.get("entities") or []]
        ),
        "bound": bound,
        "unbound": unbound,
        "slot_options": slot_options,
        "entity_options": [
            {k: v for k, v in eo.items() if k in ("role", "label", "kind")}
            | {"n": len(eo.get("options") or [])}
            for eo in res.get("entity_options") or []
        ],
        "facet_options": [
            {"facet": fo.get("facet") or fo.get("key"), "n": len(fo.get("options") or [])}
            for fo in res.get("facet_options") or []
        ],
        "open_axis_facts": res.get("open_axis_facts"),
        "alternative_measures": [a.get("label") for a in res.get("alternative_measures") or []],
        "headline": res.get("headline"),
        "coverage": {
            k: (res.get("coverage") or {}).get(k) for k in ("kind", "has_data", "observation_count")
        },
        "provenance": _label(facets.get("provenance")),
        "unit": _label(facets.get("unit")),
        "stat_var_count": len((res.get("stat_vars") or {}).get("items") or []),
        "confidence": res.get("confidence"),
        "available_at_place": res.get("available_at_place"),
    }


def slim_no_data(nd: dict) -> dict:
    out = {
        k: v
        for k, v in nd.items()
        if k not in ("flagged_match", "alternate_match", "elected_stat_vars")
    }
    svs = nd.get("elected_stat_vars") or []
    if svs:
        out["elected_stat_vars"] = _cap([sv.get("label") or sv.get("dcid") for sv in svs], 3)
    for key in ("flagged_match", "alternate_match"):
        match = nd.get(key)
        if match:
            spec = digest_spec(match) if "resolved" in match else {}
            out[key] = {
                k: spec.get(k)
                for k in ("sentence", "measure_label", "provenance", "entities", "bound")
            }
    return out


def digest_part(part: dict) -> dict:
    out = {
        "status": part.get("status"),
        "variable_text": part.get("variable_text"),
        "restatement": part.get("restatement"),
        "part_sentence": part.get("rendered_sentence"),
        "operations": part.get("operations"),
        "warnings": [w.get("code") for w in part.get("warnings") or []],
    }
    match part.get("status"):
        case "definite":
            out["spec"] = digest_spec(part["spec"])
        case "candidates":
            cs = part["candidates"]
            out["ordering"] = cs.get("ordering")
            out["truncated"] = cs.get("truncated")
            out["candidates"] = [digest_spec(s) for s in cs.get("specs") or []]
        case "no_data":
            out["no_data"] = slim_no_data(part.get("no_data") or {})
        case "unsupported":
            out["unsupported"] = {k: part["unsupported"].get(k) for k in ("code", "detail")}
    return out


def digest(item: dict, result: dict) -> dict:
    out = {k: item.get(k) for k in ("id", "query", "axes", "intent", "edge")}
    body = result.get("body")
    if body is None:
        return out | {"error": result.get("error")}
    out["http_status"] = result["http_status"]
    out["kind"] = body.get("kind")
    out["status"] = body.get("status")
    out["sentence"] = body.get("rendered_sentence")
    if body.get("kind") != "resolution":
        out["detail"] = {k: body.get(k) for k in ("reason", "scope", "detail", "code", "title")}
        return out
    out["parts"] = [digest_part(p) for p in body.get("parts") or []]
    out["timing_ms"] = (body.get("diagnostics") or {}).get("timing_ms", {}).get("total")
    return out


def render_md(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        lines.append(f"## {r['id']}: {r['query']}")
        lines.append(f"- axes: `{json.dumps(r.get('axes'))}`")
        lines.append(f"- intent: {r.get('intent')}")
        if r.get("edge"):
            lines.append(f"- edge: {r['edge']}")
        if r.get("error"):
            lines.append(f"- ERROR: {r['error']}")
            lines.append("")
            continue
        lines.append(f"- kind/status: {r.get('kind')} / {r.get('status')}")
        lines.append(f"- envelope sentence: {r.get('sentence')!r}")
        if r.get("detail"):
            lines.append(f"- detail: `{json.dumps(r['detail'])}`")
        for i, p in enumerate(r.get("parts") or []):
            lines.append(
                f"- part {i} [{p['status']}] variable_text={p['variable_text']!r} "
                f"warnings={p['warnings']}"
            )
            lines.append(f"  - restatement: `{json.dumps(p['restatement'])}`")
            if p.get("part_sentence"):
                lines.append(f"  - part sentence: {p['part_sentence']!r}")
            specs = [p["spec"]] if "spec" in p else p.get("candidates") or []
            if "candidates" in p:
                lines.append(
                    f"  - {len(specs)} candidates, ordering={p.get('ordering')}, "
                    f"truncated={p.get('truncated')}"
                )
            for j, s in enumerate(specs):
                tag = "spec" if "spec" in p else f"candidate {j}"
                lines.append(
                    f"  - {tag}: measure={s['measure_label']!r} "
                    f"provenance={s['provenance']!r} unit={s['unit']!r}"
                )
                if "candidates" in p:
                    lines.append(f"    - sentence: {s['sentence']!r}")
                lines.append(f"    - entities: {s['entities']}; bound: {s['bound']}")
                lines.append(f"    - offered to narrow: {json.dumps(s['slot_options'])}")
                if s["entity_options"] or s["facet_options"]:
                    lines.append(
                        f"    - entity_options: {s['entity_options']}; "
                        f"facet_options: {s['facet_options']}"
                    )
                lines.append(f"    - open_axis_facts: {json.dumps(s['open_axis_facts'])}")
                lines.append(f"    - alternatives: {s['alternative_measures']}")
                lines.append(
                    f"    - headline: {json.dumps(s['headline'])}; "
                    f"stat_vars: {s['stat_var_count']}; confidence: {s['confidence']}"
                )
            if p.get("no_data"):
                lines.append(f"  - no_data: `{json.dumps(p['no_data'])}`")
            if p.get("unsupported"):
                lines.append(f"  - unsupported: `{json.dumps(p['unsupported'])}`")
        lines.append("")
    return "\n".join(lines)


def load_queries(path: Path) -> list[dict]:
    items = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not items:
        raise SystemExit(f"{path} holds no queries")
    return items


def write_digests(out: Path, name: str, rows: list[dict]) -> None:
    jsonl = "".join(json.dumps(r) + "\n" for r in rows)
    (out / f"digest-{name}.jsonl").write_text(jsonl, encoding="utf-8")
    (out / f"digest-{name}.md").write_text(render_md(rows), encoding="utf-8")


def report(name: str, rows: list[dict]) -> int:
    """Print the slice line and return the number of transport errors."""
    errors = sum(1 for r in rows if r.get("error"))
    statuses = Counter(r.get("status") or r.get("kind") or "error" for r in rows)
    print(f"{name}: {len(rows)} queries, {errors} transport errors, statuses {dict(statuses)}")
    return errors


async def run_slice(
    path: Path, *, out: Path, base: str, concurrency: int, client: httpx.AsyncClient
) -> bool:
    """Run one slice and return False when every query failed at transport level."""
    items = load_queries(path)
    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> dict:
        async with sem:
            return await resolve(client, base, item["query"])

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(one(item)) for item in items]
    results = [t.result() for t in tasks]

    raw = "".join(
        json.dumps({"id": item["id"], "query": item["query"]} | res) + "\n"
        for item, res in zip(items, results, strict=True)
    )
    (out / f"raw-{path.stem}.jsonl").write_text(raw, encoding="utf-8")
    rows = [digest(item, res) for item, res in zip(items, results, strict=True)]
    write_digests(out, path.stem, rows)
    return report(path.stem, rows) < len(rows)


def redigest_slice(path: Path, *, out: Path) -> None:
    """Rebuild a slice's digests from its saved raw file without calling the resolver."""
    items = {i["id"]: i for i in load_queries(path)}
    raw_path = out / f"raw-{path.stem}.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} does not exist; run the slice before --redigest")
    raws = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    if not raws:
        raise SystemExit(f"{raw_path} holds no saved responses")
    rows = [digest(items[r["id"]], r) for r in raws]
    write_digests(out, path.stem, rows)
    print(f"{path.stem}: re-digested {len(rows)} queries")


def slice_path(arg: str) -> Path:
    """Resolve a slice name (`crs`) or a path to a queries file."""
    path = Path(arg)
    if path.suffix == ".yaml":
        return path
    return QUERIES_DIR / f"{arg}.yaml"


async def run(args: argparse.Namespace) -> int:
    paths = [slice_path(s) for s in args.slices] or sorted(QUERIES_DIR.glob("*.yaml"))
    if not paths:
        raise SystemExit(f"no queries files in {QUERIES_DIR}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.redigest:
        for path in paths:
            redigest_slice(path, out=args.out)
        return 0
    failed = []
    async with httpx.AsyncClient() as client:
        for path in paths:
            ok = await run_slice(
                path, out=args.out, base=args.base, concurrency=args.concurrency, client=client
            )
            if not ok:
                failed.append(path.stem)
    if failed:
        print(f"every query failed at transport level in: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "slices",
        nargs="*",
        help="slice names or queries-file paths (default: every file in scripts/sentence_panel/queries)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Search API base URL")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--redigest",
        action="store_true",
        help="rebuild the digests in --out from saved raw files without calling the resolver",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
