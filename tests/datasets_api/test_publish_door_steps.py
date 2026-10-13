"""The publish door's `steps`: their shape, the structure the composer walks, their own bounds,
and the facts a republish must repeat.

The door stores any `op` string. It refuses a malformed step, a repeated id, an input naming
neither an earlier step nor a version, a compare whose `sides` are missing or keyed by other
roles than its inputs, a version named anywhere but a `retrieve` step's `input`, a list
naming no version that way, a `continue_series` step whose params state no method or no pair
of periods, a list past `MAX_STEPS` entries or `MAX_STEPS_BYTES` bytes of canonical JSON, and
a value JSON cannot carry, each as `request_invalid` (400) with a pointer into `steps`, and
writes nothing. A republish of identical frame bytes whose steps differ in a fact (inputs,
operations, policies, parameters, units or counts) is refused as `publish_conflict` (409),
even when the record's builder fails. A republish that differs only in labels is accepted.
"""

from __future__ import annotations

import json

import pytest
import rfc8785
from dfo_descriptions import StepsError, StepsErrorReason

from datasets_api import publish_models, version_descriptions
from datasets_api.publish_models import MAX_STEPS, MAX_STEPS_BYTES

from ._description_doubles import (
    AUTH_HEADERS,
    DENOMINATOR,
    NUMERATOR,
    OCV,
    OPS_ID,
    datasets_client,
    publish,
    publish_body,
    quotient_steps,
    read_record,
    served_steps,
    version_rows,
)
from ._version_record_store import FakeVersionRecordDatasetStore


def _retrieve(step_id: str, version: tuple[str, str] = NUMERATOR) -> dict[str, object]:
    return {
        "id": step_id,
        "op": "retrieve",
        "input": {"id": version[0], "content_version": version[1]},
        "options": {},
    }


# Retrieves both of `publish_body`'s inputs.
_RETRIEVED = [_retrieve("r1"), _retrieve("r2", DENOMINATOR)]

_COMPARE = {"id": "c3", "op": "compare", "inputs": {"left": "r1", "right": "r2"}}


def _refused(response, *, store: FakeVersionRecordDatasetStore) -> dict:
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"].startswith("/steps")
    assert version_rows(store, OPS_ID) == set()
    return body


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(quotient_steps(), id="control-the-designs-quotient-chain"),
        pytest.param([_retrieve("r1")], id="control-one-retrieve"),
        pytest.param(
            [*_RETRIEVED, {"id": "s2", "op": "deflate_twice", "input": "r1"}],
            id="control-an-operation-no-list-holds",
        ),
        pytest.param(
            [
                _retrieve("r1"),
                {
                    "id": "s2",
                    "op": "continue_series",
                    "params": {"method": "Estimated", "periods": ["2025", "2026"]},
                },
            ],
            id="control-a-continuation-stating-its-method-and-periods",
        ),
    ],
)
def test_well_formed_steps_publish(steps: list[dict[str, object]]) -> None:
    store = FakeVersionRecordDatasetStore()

    response = publish(datasets_client(store), publish_body(steps=steps))

    assert response.status_code == 200, response.text
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert served_steps(record) == steps


@pytest.mark.parametrize(
    ("steps", "pointer"),
    [
        pytest.param([], "/steps", id="no-steps"),
        pytest.param([{"id": "s1", "op": "aggregate"}], "/steps", id="no-retrieve"),
        pytest.param(
            [
                {
                    "id": "r1",
                    "op": "retrieve",
                    "inputs": {"source": {"id": NUMERATOR[0], "content_version": NUMERATOR[1]}},
                }
            ],
            "/steps/0/inputs/source",
            id="a-version-under-a-retrieves-inputs",
        ),
        pytest.param([_retrieve("s1"), _retrieve("s1")], "/steps/1/id", id="a-repeated-id"),
        pytest.param(
            [
                *_RETRIEVED,
                {
                    "id": "s3",
                    "op": "quotient",
                    "inputs": {
                        "numerator": "r1",
                        "denominator": {"id": DENOMINATOR[0], "content_version": DENOMINATOR[1]},
                    },
                },
            ],
            "/steps/2/inputs/denominator",
            id="a-version-under-a-step-other-than-retrieve",
        ),
        pytest.param(
            [{"id": "s1", "op": "growth", "input": "s2"}, _retrieve("s2")],
            "/steps/0/input",
            id="an-input-naming-a-later-step",
        ),
        pytest.param(
            [_retrieve("s1"), {"id": "s2", "op": "share", "inputs": {"part": "s1", "total": "s9"}}],
            "/steps/1/inputs/total",
            id="an-input-naming-no-step",
        ),
        pytest.param(
            [{"id": "s1", "op": "growth", "input": "s1"}],
            "/steps/0/input",
            id="a-step-naming-itself",
        ),
        pytest.param(
            [
                _retrieve("s1"),
                {"id": "s2", "op": "growth", "input": "s3"},
                {"id": "s3", "op": "growth", "input": "s2"},
            ],
            "/steps/1/input",
            id="a-cycle-of-two-steps",
        ),
        pytest.param(
            [_retrieve("s1"), {"id": "s2", "input": "s1"}], "/steps/1/op", id="a-step-without-an-op"
        ),
        pytest.param(
            [_retrieve("s1"), {"id": "s2", "op": "growth", "input": "s1", "inputs": {"x": "s1"}}],
            "/steps/1",
            id="a-step-with-input-and-inputs",
        ),
        pytest.param(
            [*_RETRIEVED, _COMPARE],
            # An absent member points at the step that lacks it.
            "/steps/2",
            id="a-compare-without-sides",
        ),
        pytest.param(
            [*_RETRIEVED, {**_COMPARE, "sides": {"left": {}, "other": {}}}],
            "/steps/2/sides",
            id="a-compare-whose-sides-name-other-roles",
        ),
        pytest.param([*_RETRIEVED, {**_COMPARE, "sides": {}}], "/steps/2/sides", id="empty-sides"),
        pytest.param(
            [{**_retrieve("s1"), "kept": {"entities": -1}}],
            "/steps/0/kept/entities",
            id="a-negative-count",
        ),
        *(
            pytest.param(
                [_retrieve("s1"), {"id": "s2", "op": "continue_series", **params}],
                pointer,
                id=f"a-continuation-{name}",
            )
            for name, params, pointer in [
                # An absent member points at the step that lacks it.
                ("without-params", {}, "/steps/1"),
                ("with-empty-params", {"params": {}}, "/steps/1/params"),
                (
                    "whose-method-is-a-list",
                    {"params": {"method": [], "periods": ["1", "2"]}},
                    "/steps/1/params",
                ),
                (
                    "with-an-empty-method",
                    {"params": {"method": "", "periods": ["1", "2"]}},
                    "/steps/1/params",
                ),
                ("without-periods", {"params": {"method": "Estimated"}}, "/steps/1/params"),
                (
                    "with-one-period",
                    {"params": {"method": "Estimated", "periods": ["2025"]}},
                    "/steps/1/params",
                ),
            ]
        ),
    ],
)
def test_malformed_steps_are_refused_with_a_pointer_and_write_nothing(
    steps: list[dict[str, object]], pointer: str
) -> None:
    store = FakeVersionRecordDatasetStore()

    body = _refused(publish(datasets_client(store), publish_body(steps=steps)), store=store)

    assert body["pointer"] == pointer


def test_a_chained_derivation_publishes_with_its_roots_as_inputs() -> None:
    """dfo-ai lists the immediate input frames, here a filtered frame by the `cv_` of its
    bytes, while its steps retrieve the root datasets. The record's inputs are the roots."""
    store = FakeVersionRecordDatasetStore()
    filtered = "cv_" + "f0" * 16
    steps = [
        _retrieve("s1"),
        {"id": "s2", "op": "filter", "input": "s1", "params": {}},
        _retrieve("s3", DENOMINATOR),
        {"id": "s4", "op": "quotient", "inputs": {"numerator": "s2", "denominator": "s3"}},
    ]
    body = publish_body(steps=steps) | {"input_content_versions": [filtered, DENOMINATOR[1]]}

    response = publish(datasets_client(store), body)

    assert response.status_code == 200, response.text
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert record.computed is not None
    assert [(held.id, held.content_version) for held in record.computed.inputs] == [
        NUMERATOR,
        DENOMINATOR,
    ]


def test_a_step_list_at_its_count_bound_publishes() -> None:
    store = FakeVersionRecordDatasetStore()
    steps = [*_RETRIEVED, *(_retrieve(f"s{index}") for index in range(MAX_STEPS - 2))]

    response = publish(datasets_client(store), publish_body(steps=steps))

    assert response.status_code == 200, response.text


def test_a_step_list_past_its_count_bound_is_refused_naming_the_bound() -> None:
    store = FakeVersionRecordDatasetStore()
    steps = [_retrieve(f"s{index}") for index in range(MAX_STEPS + 1)]

    body = _refused(publish(datasets_client(store), publish_body(steps=steps)), store=store)

    assert body["pointer"] == "/steps"
    assert f"MAX_STEPS={MAX_STEPS}" in body["detail"]


def _canonical_size(steps: list[dict[str, object]]) -> int:
    return len(rfc8785.dumps(steps))


def test_a_step_list_past_its_byte_bound_is_refused_naming_the_bound() -> None:
    """Aggregate steps with full period coverage pass the byte bound well inside the count
    bound, so the byte bound is the one broken."""
    store = FakeVersionRecordDatasetStore()
    coverage = [{"entities": 54, "period": str(1525 + offset)} for offset in range(500)]
    steps: list[dict[str, object]] = [_retrieve("s0")]
    while _canonical_size(steps) <= MAX_STEPS_BYTES:
        steps.append(
            {"id": f"a{len(steps)}", "op": "aggregate", "input": "s0", "period_coverage": coverage}
        )
    assert len(steps) <= MAX_STEPS, "the byte bound must be reachable inside the count bound"

    body = _refused(publish(datasets_client(store), publish_body(steps=steps)), store=store)

    assert body["pointer"] == "/steps"
    assert f"MAX_STEPS_BYTES={MAX_STEPS_BYTES}" in body["detail"]


def test_a_whole_list_rule_other_than_the_byte_bound_is_refused_under_its_own_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole-list refusal the kit gives for a rule other than the byte bound, here its count
    bound, names that rule."""

    def too_many(steps: object) -> None:
        raise StepsError(reason=StepsErrorReason.TOO_MANY_STEPS, step=None, detail="65 steps")

    monkeypatch.setattr(publish_models, "check_steps", too_many)
    store = FakeVersionRecordDatasetStore()

    body = _refused(publish(datasets_client(store), publish_body(steps=_RETRIEVED)), store=store)

    assert body["pointer"] == "/steps"
    assert "too_many_steps" in body["detail"]
    assert "MAX_STEPS_BYTES" not in body["detail"]


def _steps_of_canonical_size(size: int) -> list[dict[str, object]]:
    """Steps whose `params` hold 1e-07 a thousand times, which Python writes in five bytes
    and RFC 8785 in four, padded with a string to exactly `size` canonical bytes."""
    rates = [1e-07] * 1000
    steps: list[dict[str, object]] = [
        _retrieve("s0"),
        {"id": "f1", "op": "filter", "input": "s0", "params": {"pad": "", "rates": rates}},
    ]
    pad = size - _canonical_size(steps)
    assert pad >= 0
    steps[1]["params"] = {"pad": "x" * pad, "rates": rates}
    assert _canonical_size(steps) == size
    return steps


def test_a_step_list_is_counted_in_canonical_json_at_the_byte_bound() -> None:
    """Python spells 1e-07 in one byte more than RFC 8785 does, so the list at the bound is
    over it when counted with `json.dumps` and at it when counted as the kit counts."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    at_bound = _steps_of_canonical_size(MAX_STEPS_BYTES)
    assert len(json.dumps(at_bound, separators=(",", ":")).encode()) > MAX_STEPS_BYTES

    accepted = publish(client, publish_body(steps=at_bound))
    refused = publish(client, publish_body(steps=_steps_of_canonical_size(MAX_STEPS_BYTES + 1)))

    assert accepted.status_code == 200, accepted.text
    assert refused.status_code == 400
    assert refused.json()["pointer"] == "/steps"
    assert f"MAX_STEPS_BYTES={MAX_STEPS_BYTES}" in refused.json()["detail"]


@pytest.mark.parametrize(
    "unwritable",
    [
        pytest.param("\ud800", id="a-lone-surrogate"),
        pytest.param(2**53, id="an-integer-past-2-to-the-53"),
    ],
)
def test_a_value_json_cannot_carry_is_refused_as_not_i_json(unwritable: object) -> None:
    store = FakeVersionRecordDatasetStore()
    steps: list[dict[str, object]] = [
        _retrieve("s0"),
        {"id": "f1", "op": "filter", "input": "s0", "params": {"x": unwritable}},
    ]
    # `ensure_ascii` writes the lone surrogate as an escape, since httpx's `json=` cannot encode it.
    content = json.dumps(publish_body(steps=steps))

    response = datasets_client(store).post(
        "/api/datasets/frames/publish",
        content=content,
        headers={**AUTH_HEADERS, "content-type": "application/json"},
    )

    body = _refused(response, store=store)
    assert body["pointer"] == "/steps"


@pytest.mark.parametrize(
    ("changed", "status", "code"),
    [
        pytest.param(quotient_steps(kept_entities=37), 409, "publish_conflict", id="a-kept-count"),
        pytest.param(
            [*quotient_steps()[:2], {**quotient_steps()[2], "policy": "quotient_of_union"}],
            409,
            "publish_conflict",
            id="a-policy",
        ),
        pytest.param(
            [*quotient_steps()[:2], {**quotient_steps()[2], "op": "share"}],
            409,
            "publish_conflict",
            id="an-operation",
        ),
        pytest.param(quotient_steps()[:2], 409, "publish_conflict", id="a-missing-step"),
        pytest.param(
            quotient_steps(numerator_label="ODA, gross disbursements"),
            200,
            None,
            id="control-a-label-alone",
        ),
    ],
)
def test_a_republish_of_identical_bytes_with_other_step_facts_is_refused(
    changed: list[dict[str, object]], status: int, code: str | None
) -> None:
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    steps = quotient_steps()
    assert publish(client, publish_body(steps=steps)).status_code == 200

    response = publish(client, publish_body(steps=changed))

    assert response.status_code == status, response.text
    if code is not None:
        assert response.json()["code"] == code
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert served_steps(record) == steps


def test_a_contradicting_republish_is_refused_when_the_record_builder_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step comparison needs no record built, so a builder defect leaves the version's
    stored record to refuse steps that contradict it."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    steps = quotient_steps()
    assert publish(client, publish_body(steps=steps)).status_code == 200

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("builder defect")

    monkeypatch.setattr(version_descriptions, "publish_record", broken)
    response = publish(client, publish_body(steps=quotient_steps(kept_entities=37)))

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "publish_conflict"
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert served_steps(record) == steps
