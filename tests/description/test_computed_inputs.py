"""A computed version's inputs whose ids are unknown, as served by `describe_version`.

A version published without steps keeps each input's content version alone. Its description
serves the input with `id: null` in `freshness.inputs` and `cite.inputs`, and with `ref: null`
and no `description` link in `sources`, each with a gap. The label is the one stored with the
input, else the content version with a gap.
"""

from __future__ import annotations

import pytest

from description.facts import ComputedFacts, InputVersion

from .test_accept_compose import _GNI, _ODA, _describe_version, _quotient_record


def _without_ids(*, labelled: bool) -> list[InputVersion]:
    return [
        InputVersion(
            id=None,
            content_version=version.content_version,
            **({"label": version.label} if labelled else {}),
        )
        for version in (_ODA, _GNI)
    ]


def test_control_inputs_with_ids_serve_their_ids_and_links_without_gaps() -> None:
    served = _describe_version(_quotient_record())

    assert [entry["id"] for entry in served["freshness"]["inputs"]] == [_ODA.id, _GNI.id]
    assert all("description" in source for source in served["sources"])
    assert not [gap for gap in served["gaps"] if gap.endswith((".id", ".ref"))]


@pytest.mark.parametrize(
    ("labelled", "labels", "label_gaps"),
    [
        pytest.param(
            True, [(_ODA.label, True), (_GNI.label, True)], set(), id="labels-stored-with-inputs"
        ),
        pytest.param(
            False,
            [(_ODA.content_version, None), (_GNI.content_version, None)],
            {"freshness.inputs.0.label", "sources.1.label", "cite.inputs.1.label"},
            id="no-labels",
        ),
    ],
)
def test_inputs_without_ids_serve_null_ids_with_gaps(
    labelled: bool, labels: list[tuple[str, bool | None]], label_gaps: set[str]
) -> None:
    record = _quotient_record().model_copy(
        update={"computed": ComputedFacts(inputs=_without_ids(labelled=labelled), composed=None)}
    )

    served = _describe_version(record)

    inputs = served["freshness"]["inputs"]
    assert [(entry["id"], entry["content_version"]) for entry in inputs] == [
        (None, _ODA.content_version),
        (None, _GNI.content_version),
    ]
    assert [(entry["label"], entry.get("at_mint")) for entry in inputs] == labels
    assert [entry["id"] for entry in served["cite"]["inputs"]] == [None, None]
    assert [
        (source["ref"], source["content_version"], "description" in source)
        for source in served["sources"]
    ] == [(None, _ODA.content_version, False), (None, _GNI.content_version, False)]
    gaps = set(served["gaps"])
    for n in (0, 1):
        assert {f"freshness.inputs.{n}.id", f"cite.inputs.{n}.id", f"sources.{n}.ref"} <= gaps
    assert label_gaps <= gaps
