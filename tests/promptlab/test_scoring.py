"""Tests for nomination counts, alternate recall, and the seed-flip rate over stored calls."""

from __future__ import annotations

import pytest

from promptlab.bindmenu import MenuEntry
from promptlab.expectations import LegExpectation
from promptlab.scoring import make_nomination_correctness, seed_flip_rate
from tests.conftest import base_response, minimal_spec


def _expected(status: str, *stat_vars: str) -> LegExpectation:
    return LegExpectation(statuses=(status,), shapes=(), stat_vars=stat_vars)


_CANDIDATES = _expected("candidates", "sv/Alpha", "sv/Beta")
_DEFINITE = _expected("definite")
_NO_DATA = _expected("no_data")


def _entry(
    *, index: int, label: str, five_tuple: dict, sample_dcids: tuple[str, ...] = ()
) -> MenuEntry:
    return MenuEntry(
        index=index,
        label=label,
        five_tuple=five_tuple,
        sample_dcids=sample_dcids,
        retrieval_score=1.0,
    )


def _arcs(*, pop_type: str, meas_prop: str, stat_type: str = "measuredValue") -> dict:
    """Build node arcs for the three required shape fields."""
    return {
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": stat_type}]},
    }


def _five_tuple(
    *,
    pop_type: str,
    meas_prop: str,
    stat_type: str = "measuredValue",
    denominator: str | None = None,
) -> dict:
    """Build the menu-side shape tuple."""
    return {
        "population_type": pop_type,
        "measured_property": meas_prop,
        "stat_type": stat_type,
        "measurement_qualifier": None,
        "measurement_denominator": denominator,
    }


class _StubGraph:
    """Map dcids to node arcs; absent dcids return None."""

    def __init__(self, arcs_by_dcid: dict[str, dict | None] | None = None):
        self._arcs_by_dcid = arcs_by_dcid or {}
        self.node_arcs_calls: list[str] = []

    def node_arcs(self, dcid: str) -> dict | None:
        self.node_arcs_calls.append(dcid)
        return self._arcs_by_dcid.get(dcid)


class _RaisingGraph:
    def node_arcs(self, dcid: str) -> dict | None:
        raise RuntimeError("graph error: network unavailable")


_MENU = (
    _entry(index=0, label="Alpha shape", five_tuple=_five_tuple(pop_type="A", meas_prop="a")),
    _entry(index=1, label="Beta shape", five_tuple=_five_tuple(pop_type="B", meas_prop="b")),
)

_GRAPH_ARCS = {
    "sv/Alpha": _arcs(pop_type="A", meas_prop="a"),
    "sv/Beta": _arcs(pop_type="B", meas_prop="b"),
}


def test_emits_nothing_for_no_data_case():
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    result = nomination_correctness(output={"also_plausible": [1]}, expected=_NO_DATA, menu=_MENU)
    assert result == []


def test_emits_nothing_for_unsupported_case():
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    result = nomination_correctness(
        output={"also_plausible": None},
        expected=_expected("unsupported"),
        menu=_MENU,
    )
    assert result == []


def test_definite_case_emits_over_nomination_count():
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": [1]},
        expected=_DEFINITE,
        menu=_MENU,
    )
    assert len(result) == 1
    ev = result[0]
    assert ev.name == "over_nomination_count"
    assert ev.value == 1


def test_definite_case_zero_nominations():
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": None},
        expected=_DEFINITE,
        menu=_MENU,
    )
    assert result[0].value == 0


def test_candidates_case_full_recall_via_five_tuple():
    """Index 0 is chosen; nominated index 1 gives recall 1/2 over both expected StatVars."""
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": [1]},
        expected=_CANDIDATES,
        menu=_MENU,
    )
    assert len(result) == 1
    ev = result[0]
    assert ev.name == "expected_alternate_recall"
    assert abs(ev.value - 0.5) < 1e-9
    assert "1/2" in (ev.comment or "")


def test_candidates_case_both_alternates_in_also_plausible():
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": [0, 1]},
        expected=_CANDIDATES,
        menu=_MENU,
    )
    assert result[0].value == 1.0


def test_candidates_case_neither_alternate_nominated():
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": None},
        expected=_CANDIDATES,
        menu=_MENU,
    )
    assert result[0].value == 0.0


def test_candidates_case_falls_back_to_sample_dcids_when_five_tuple_unresolvable():
    """sample_dcids maps Gamma to its menu entry when graph lookup returns None."""
    menu = (
        _entry(
            index=0,
            label="Gamma shape",
            five_tuple=_five_tuple(pop_type="G", meas_prop="g"),
            sample_dcids=("sv/Gamma", "sv/OtherMember"),
        ),
    )
    graph = _StubGraph({"sv/Gamma": None})
    nomination_correctness = make_nomination_correctness(graph)
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": []},
        expected=_expected("candidates", "sv/Gamma"),
        menu=menu,
    )
    # Gamma resolves to chosen index 0, which is absent from also_plausible: a scored miss.
    assert result[0].name == "expected_alternate_recall"
    assert result[0].value == 0.0


def test_candidates_case_unscorable_when_no_expected_alternate_resolves():
    graph = _StubGraph({})
    nomination_correctness = make_nomination_correctness(graph)
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": [1]},
        expected=_CANDIDATES,
        menu=_MENU,
    )
    assert len(result) == 1
    ev = result[0]
    assert ev.name == "expected_alternate_recall"
    assert ev.value == "unscorable"
    assert ev.data_type == "CATEGORICAL"


def test_candidates_case_partial_unscorable_reported_not_scored_as_miss():
    """Recall uses the one resolvable alternate (1/1) and reports the other as unscorable."""
    graph = _StubGraph({"sv/Alpha": _GRAPH_ARCS["sv/Alpha"]})  # sv/Beta absent
    nomination_correctness = make_nomination_correctness(graph)
    result = nomination_correctness(
        output={"chosen_shape_index": 1, "also_plausible": [0]},
        expected=_CANDIDATES,
        menu=_MENU,
    )
    ev = result[0]
    assert ev.value == 1.0
    assert "unscorable" in (ev.comment or "")
    assert "sv/Beta" in (ev.comment or "")


def test_candidates_case_denominator_unresolved_resolves_via_sentinel():
    """A ProvisionalNode denominator matches the unresolved menu entry."""
    menu = (
        _entry(
            index=0,
            label="Resolved-denominator shape",
            five_tuple=_five_tuple(
                pop_type="EconomicActivity",
                meas_prop="amount",
                denominator="PerCapita",
            ),
        ),
        _entry(
            index=1,
            label="Unresolved-denominator shape",
            five_tuple={
                "population_type": "EconomicActivity",
                "measured_property": "amount",
                "stat_type": "measuredValue",
                "measurement_qualifier": None,
                "measurement_denominator": "(present, unresolved)",
            },
        ),
    )
    arcs = {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementDenominator": {"nodes": [{"dcid": "PerCapita", "types": ["ProvisionalNode"]}]},
    }
    graph = _StubGraph({"sv/Target": arcs})
    nomination_correctness = make_nomination_correctness(graph)
    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": [1]},
        expected=_expected("candidates", "sv/Target"),
        menu=menu,
    )
    # Both shapes carry the PerCapita dcid; the unresolved marker selects index 1.
    assert result[0].name == "expected_alternate_recall"
    assert result[0].value == 1.0


def test_graph_errors_propagate():
    nomination_correctness = make_nomination_correctness(_RaisingGraph())
    with pytest.raises(RuntimeError, match="graph error"):
        nomination_correctness(
            output={"chosen_shape_index": 0, "also_plausible": [1]},
            expected=_CANDIDATES,
            menu=_MENU,
        )


def test_a_leg_accepting_several_statuses_is_unscored():
    nomination_correctness = make_nomination_correctness(_StubGraph(_GRAPH_ARCS))
    expected = LegExpectation(
        statuses=("definite", "candidates"), shapes=(), stat_vars=("sv/Alpha",)
    )

    result = nomination_correctness(
        output={"chosen_shape_index": 0, "also_plausible": [1]}, expected=expected, menu=_MENU
    )

    assert result == []


# ---------------------------------------------------------------------------
# seed_flip_rate
# ---------------------------------------------------------------------------


def _definite_response(spec_id: str) -> dict:
    return base_response(
        status="definite", parts=[{"status": "definite", "spec": minimal_spec(spec_id)}]
    )


def _no_data_response(reason: str) -> dict:
    return base_response(
        status="no_data", parts=[{"status": "no_data", "no_data": {"reason": reason}}]
    )


@pytest.mark.parametrize(
    "outputs",
    [
        pytest.param([], id="zero-outputs"),
        pytest.param([_definite_response("s1")], id="single-seed"),
    ],
)
def test_seed_flip_rate_is_unscored_without_multiple_seeds(outputs):
    assert seed_flip_rate(outputs) is None


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        pytest.param(
            [_definite_response("s1")] * 3,
            0.0,
            id="unanimous",
        ),
        pytest.param(
            [_definite_response("s1"), _definite_response("s1"), _definite_response("s2")],
            1 - 2 / 3,
            id="two-of-three",
        ),
        pytest.param(
            [_definite_response("s1"), _definite_response("s2"), _definite_response("s3")],
            1 - 1 / 3,
            id="three-way-tie",
        ),
        pytest.param(
            [
                _definite_response("s1"),
                _no_data_response("no_observations"),
                _definite_response("s1"),
            ],
            1 - 2 / 3,
            id="different-statuses-never-share-an-answer",
        ),
    ],
)
def test_seed_flip_rate_is_the_share_outside_the_modal_answer(outputs, expected):
    assert seed_flip_rate(outputs) == pytest.approx(expected)
