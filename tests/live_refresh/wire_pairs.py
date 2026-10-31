"""Record request and response pairs of the live-refresh routes from the composed app.

`record_wire_pairs` drives one scripted run and returns each named exchange: the component
read, the recipe routes, the frame refresh in each of its answers, the two render routes, and
a publish carrying a recipe. `wire_pairs.json` beside this module is that recording. A client of
these routes in another service checks its mock against the file, and
`test_live_refresh_chain.py` fails when the file no longer matches what the app answers.

Values that differ between runs or builds are replaced, the same value by the same stand-in
everywhere, so the recording is stable and ids still line up across pairs:

- content versions, derived frame ids, recipe hashes, replay keys, operation ids, component
  ids, description vocabulary revisions, and artifact digests are numbered in order of first appearance, in their own grammar
  (`cv_00000000000000000000000000000001`);
- times read `<timestamp>`, dates `<date>`, a citation's year `<year>`, and `trace_id` and
  `capability_version` their own name in angle brackets;
- the principal header holds `<LIVE_REFRESH_PRINCIPAL>` or `<DATASETS_FIRST_PARTY_PRINCIPAL>`.

A JSON body is recorded whole. A file body is recorded as its first bytes: text for CSV, hex
for a ZIP. Record again from the repository root:

    SEARCH_GRAPH_BASE=https://graph.offline.invalid \
        uv run --frozen --extra engine --extra eval --extra datasets --extra graph \
        python -m tests.live_refresh.wire_pairs

`SEARCH_GRAPH_BASE` is the offline value tests/conftest.py sets; the module does not import
without it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from tests.live_refresh.composed_app import (
    DATASET,
    FRAME_OPTIONS,
    GROUPBY,
    LIVE_REFRESH_PRINCIPAL,
    PRINCIPAL_HEADER,
    PUBLISH_PRINCIPAL,
    RECIPIENTS,
    ComposedApp,
    compose,
    derived_publish_body,
    offline_services,
)

WIRE_PAIRS_PATH = Path(__file__).with_name("wire_pairs.json")

_GHANA, _KENYA, _TANZANIA, _UGANDA = RECIPIENTS
_COMPONENTS = "/api/presentation/components"
_REFRESH = "/api/datasets/frames/refresh"
_PRINCIPAL_NAMES = {
    LIVE_REFRESH_PRINCIPAL: "<LIVE_REFRESH_PRINCIPAL>",
    PUBLISH_PRINCIPAL: "<DATASETS_FIRST_PARTY_PRINCIPAL>",
}
_RESPONSE_HEADERS = ("content-type", "cache-control", "content-disposition")
_TEXT_BODY_CHARS = 240
_BINARY_BODY_BYTES = 4

# Ids numbered in order of first appearance: the pattern, then the prefix and hex width of the
# stand-in. A longer prefix comes before a prefix it ends with (`ocv_` before `cv_`).
_NUMBERED_IDS = (
    (re.compile(r"\bocv_[0-9a-f]{32}\b"), "ocv_", 32),
    (re.compile(r"\bcv_[0-9a-f]{32}\b"), "cv_", 32),
    (re.compile(r"\bops_[0-9a-f]{32}\b"), "ops_", 32),
    (re.compile(r"\brnf_[0-9a-f]{32}\b"), "rnf_", 32),
    (re.compile(r"\brpk_[0-9a-f]{32}\b"), "rpk_", 32),
    (re.compile(r"\bop_[0-9a-f]{64}\b"), "op_", 64),
    (re.compile(r"\bcmp_[0-9a-f]{24}\b"), "cmp_", 24),
    (re.compile(r"\bvoc_[0-9a-f]{32}\b"), "voc_", 32),
    (re.compile(r"\bsha256:[0-9a-f]{64}\b"), "sha256:", 64),
)
_NAMED_VALUES = (
    (re.compile(r'"trace_id": "[0-9a-f]+"'), '"trace_id": "<trace_id>"'),
    (
        re.compile(r'"capability_version": "[a-z]+_[0-9a-f]+"'),
        '"capability_version": "<capability_version>"',
    ),
    (
        re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)?"),
        "<timestamp>",
    ),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
    (re.compile(r'("citation": "[^"]*\()\d{4}(\))'), r"\1<year>\2"),
)
_ABOUT = (
    "Request and response pairs of the dfo-apis live-refresh routes, recorded from the "
    "composed test app by tests/live_refresh/wire_pairs.py. Ids are numbered stand-ins in "
    "their own grammar, the same id by the same stand-in in every pair. <timestamp>, <date>, "
    "a citation's <year>, <trace_id>, <capability_version>, and the two principal names stand "
    "for values that differ between runs or deployments."
)


def _stable(recording: object) -> object:
    """`recording` with every run-specific value replaced by its stand-in."""
    text = json.dumps(recording)
    for pattern, prefix, width in _NUMBERED_IDS:
        numbers: dict[str, str] = {}

        def stand_in(
            match: re.Match[str],
            numbers: dict[str, str] = numbers,
            prefix: str = prefix,
            width: int = width,
        ) -> str:
            return numbers.setdefault(match.group(), f"{prefix}{len(numbers) + 1:0{width}x}")

        text = pattern.sub(stand_in, text)
    for pattern, replacement in _NAMED_VALUES:
        text = pattern.sub(replacement, text)
    return json.loads(text)


def _response_body(resp: httpx.Response) -> dict[str, object]:
    content_type = resp.headers.get("content-type", "")
    if "json" in content_type:
        return {"body": resp.json()}
    if content_type.startswith("text/"):
        return {"body_starts": resp.text[:_TEXT_BODY_CHARS]}
    return {"body_starts_hex": resp.content[:_BINARY_BODY_BYTES].hex()}


class _Recorder:
    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.pairs: dict[str, object] = {}

    def call(
        self,
        name: str | None,
        method: str,
        path: str,
        *,
        principal: str | None = None,
        params: Mapping[str, str] | None = None,
        body: object = None,
    ) -> httpx.Response:
        """Send one request, and record the exchange under `name` unless it is `None`."""
        headers = {} if principal is None else {PRINCIPAL_HEADER: principal}
        resp = self._client.request(method, path, params=params, json=body, headers=headers)
        if name is None:
            return resp
        sent_headers = {} if principal is None else {PRINCIPAL_HEADER: _PRINCIPAL_NAMES[principal]}
        request: dict[str, object] = {
            "method": method,
            "path": str(resp.request.url.raw_path, "ascii"),
        }
        if body is not None:
            sent_headers["content-type"] = "application/json"
            request["body"] = body
        request["headers"] = sent_headers
        self.pairs[name] = {
            "request": request,
            "response": {
                "status": resp.status_code,
                "headers": {
                    header: resp.headers[header]
                    for header in _RESPONSE_HEADERS
                    if header in resp.headers
                },
                **_response_body(resp),
            },
        }
        return resp


def _frame(resp: httpx.Response) -> dict[str, str]:
    assert resp.status_code == 200, resp.text
    return resp.json()["frame"]


def record_wire_pairs(*, composed: ComposedApp, client: TestClient) -> dict[str, object]:
    """Run the scripted exchanges against a freshly composed app and return the recording.

    `client` is an entered `TestClient` over `composed.app`, so the surfaces' lifespans ran.
    """
    recorder = _Recorder(client)
    call = recorder.call
    live, publish = LIVE_REFRESH_PRINCIPAL, PUBLISH_PRINCIPAL
    drawn = [_GHANA, _KENYA, _TANZANIA]
    left_out = [{"root": 0, "coordinates": {"recipient": _UGANDA}}]
    window = {"start": {"anchor": "latest", "offset": -1}, "end": {"anchor": "latest", "offset": 0}}

    minted = call(None, "POST", "/api/datasets", body=composed.mint_body())
    assert minted.status_code == 200, minted.text
    stored = {"spec_id": DATASET, "content_version": minted.json()["content_version"]}
    created = call(
        None,
        "POST",
        _COMPONENTS,
        principal=publish,
        body={
            "intent": {
                "type": "line_chart",
                "frame_ref": stored,
                "select": {"stat_var": "dcid:LiveRefreshGrants", "recipient": [_KENYA, _GHANA]},
            }
        },
    )
    assert created.status_code == 201, created.text
    card = created.json()["component_id"]
    card_render = f"{_COMPONENTS}/{card}/revisions/1/render"

    call("component_read.at_revision", "GET", f"{_COMPONENTS}/{card}", params={"revision": "1"})
    call("frame_refresh.current", "POST", _REFRESH, principal=live, body=stored)
    call("frame_refresh.refresh_not_permitted", "POST", _REFRESH, body=stored)

    derived_body = derived_publish_body(
        root=stored,
        rows=composed.observations.growth_rows(drawn),
        anchor={"period": "2024", "rule": "half1", "left_out": left_out},
    )
    published = call(
        "publish.with_recipe",
        "POST",
        "/api/datasets/frames/publish",
        principal=publish,
        body=derived_body,
    )
    assert published.status_code == 200, published.text
    derived = {key: derived_body[key] for key in ("spec_id", "content_version")}
    recipe_path = "/api/datasets/frames/{spec_id}/{content_version}/recipe"
    call("recipe_read.found", "GET", recipe_path.format(**derived), principal=publish)
    call("frame_refresh.frame_not_refreshable", "POST", _REFRESH, principal=live, body=derived)
    derived_created = call(
        None,
        "POST",
        _COMPONENTS,
        principal=publish,
        body={
            "intent": {
                "type": "line_chart",
                "frame_ref": derived,
                "select": {"recipient": list(RECIPIENTS)},
                "window": window,
            }
        },
    )
    assert derived_created.status_code == 201, derived_created.text
    derived_card = derived_created.json()["component_id"]

    # A release for three of the four recipients, rebuilt through the operations worker.
    composed.set_inline_budget(inline=False)
    composed.move(add_year=2025, recipients=drawn)
    pending = call("frame_refresh.pending", "POST", _REFRESH, principal=live, body=stored)
    assert pending.json()["status"] == "pending", pending.text
    composed.run_worker()
    composed.set_inline_budget(inline=True)
    target = _frame(call("frame_refresh.moved", "POST", _REFRESH, principal=live, body=stored))

    call("render.dataset_card", "POST", card_render, principal=live, body={"frame": target})
    call(
        "render.render_target_mismatch",
        "POST",
        card_render,
        principal=live,
        body={"frame": derived},
    )
    call(
        "render_data.csv",
        "GET",
        f"{card_render}/data",
        principal=live,
        params={**target, "format": "csv", "bundle": "false"},
    )
    call("render_data.bundle", "GET", f"{card_render}/data", principal=live, params=target)

    replay_body = derived_publish_body(
        root=target,
        rows=composed.observations.growth_rows(drawn),
        anchor={"period": "2025", "rule": "half1", "left_out": left_out},
    )
    replays = "/api/datasets/recipes/replays/{}".format
    replay_key = replay_body["recipe"]["replay_key"]
    replay = {key: replay_body[key] for key in ("spec_id", "content_version")}
    call("replay_lookup.recipe_not_found", "GET", replays(replay_key), principal=publish)
    call("recipe_read.recipe_not_found", "GET", recipe_path.format(**replay), principal=publish)
    replayed = call(
        None, "POST", "/api/datasets/frames/publish", principal=publish, body=replay_body
    )
    assert replayed.status_code == 200, replayed.text
    found = call("replay_lookup.found", "GET", replays(replay_key), principal=publish)
    call(
        "render.derived_card_with_anchor_warning",
        "POST",
        f"{_COMPONENTS}/{derived_card}/revisions/1/render",
        principal=live,
        body={"frame": found.json()},
    )

    # The current frame gains a summed file while one StatVar contributes, then a second
    # StatVar contributes: the frame's rendering is ambiguous, and a new mint of the sum has a
    # version of its own.
    summed = {**FRAME_OPTIONS, "groupby": GROUPBY}
    assert call(None, "POST", "/api/datasets", body=composed.mint_body(summed)).status_code == 200
    composed.move(add_stat_var=True)
    call("frame_refresh.rendering_not_recorded", "POST", _REFRESH, principal=live, body=target)
    assert call(None, "POST", "/api/datasets", body=composed.mint_body(summed)).status_code == 200
    call("versions_list", "GET", f"/api/datasets/{DATASET}/versions")

    return {"about": _ABOUT, "pairs": _stable(recorder.pairs)}


def main() -> None:
    with offline_services():
        composed = compose()
        with TestClient(composed.app) as client:
            recording = record_wire_pairs(composed=composed, client=client)
    WIRE_PAIRS_PATH.write_text(json.dumps(recording, indent=2) + "\n")
    print(f"recorded {len(recording['pairs'])} pairs to {WIRE_PAIRS_PATH}")


if __name__ == "__main__":
    main()
