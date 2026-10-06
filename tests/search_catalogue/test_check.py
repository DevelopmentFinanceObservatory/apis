"""Pins the engine check: the pin, the points, the places it runs at."""

from __future__ import annotations

import asyncio
from typing import get_args

import pytest

from search.catalogue.check import (
    EMPTY_PREVIEW_REASONS,
    INFRA_FAILURES,
    ROLE_KEYS,
    FailedRun,
    RunFailure,
    check_example,
    check_places,
    judge,
)
from search.catalogue.examples import Example
from search.engine.errors import EngineInputError
from search.models import RESOLVE_RESPONSE_ADAPTER, ObservationsOmissionReason, ResolveResponse
from tests.conftest import base_response
from tests.search_catalogue.support import (
    Elected,
    FakeEngine,
    catalogue_answer,
    definite,
    no_data,
    omitted,
    serving,
    timed_out,
)

_EXAMPLE = Example(
    id=0, template="Population of {place}", stat_var="Count_Person", place="country/KEN"
)
_AID = "Amount_FinancialFlow_Gross_Disbursement_CRSPurposeCode72010_Nominal"


def _judge(response: ResolveResponse) -> tuple[RunFailure | None, tuple[str, ...]]:
    return judge(response, stat_var="Count_Person", place="country/KEN", role="observationAbout")


def test_a_definite_part_electing_the_pin_with_points_passes():
    assert _judge(definite("Count_Person_Female", "Count_Person")) == (
        None,
        ("Count_Person_Female", "Count_Person"),
    )


def test_the_pin_catches_a_wrong_stat_var_with_data():
    assert _judge(definite("Count_Person_Female")) == (
        "stat_var_not_elected",
        ("Count_Person_Female",),
    )


def test_an_answer_for_another_country_fails_the_scope_check():
    assert judge(
        definite("Count_Person"),
        stat_var="Count_Person",
        place="country/GEO",
        role="observationAbout",
    )[0] == ("scope_mismatch")


def test_a_preview_of_another_country_fails_even_when_the_place_resolved():
    response = definite("Count_Person", place="country/GEO").model_dump(mode="json")
    [series] = response["parts"][0]["spec"]["resolved"]["observations"]["series"]
    series["entities"] = [{"role": "observationAbout", "ref": "country/KEN"}]
    response = serving(RESOLVE_RESPONSE_ADAPTER.validate_python(response), "Count_Person")
    assert judge(
        response, stat_var="Count_Person", place="country/GEO", role="observationAbout"
    ) == ("preview_scope_mismatch", ("Count_Person",))


def test_aid_from_a_country_fails_an_example_asking_for_aid_to_it():
    reversed_ = definite(_AID, place="country/GEO", role="origin")
    assert judge(reversed_, stat_var=_AID, place="country/GEO", role="destination")[0] == (
        "scope_mismatch"
    )
    preview_reversed = definite(_AID, place="country/GEO", role="destination").model_dump(
        mode="json"
    )
    [series] = preview_reversed["parts"][0]["spec"]["resolved"]["observations"]["series"]
    series["entities"] = [{"role": "origin", "ref": "country/GEO"}]
    assert judge(
        serving(RESOLVE_RESPONSE_ADAPTER.validate_python(preview_reversed), _AID),
        stat_var=_AID,
        place="country/GEO",
        role="destination",
    )[0] == ("preview_scope_mismatch")


def test_aid_to_the_example_country_passes():
    answer = definite(_AID, place="country/GEO", role="destination")
    assert judge(answer, stat_var=_AID, place="country/GEO", role="destination") == (
        None,
        (_AID,),
    )


def test_the_check_passes_each_run_its_own_place_and_the_example_role():
    example = Example(
        id=0, template="Aid to {place}", stat_var=_AID, place="country/KEN", role="recipient"
    )
    engine = FakeEngine(
        elected={},
        answers={
            "Aid to Kenya": definite(_AID, place="country/KEN", role="destination"),
            "Aid to Georgia": definite(_AID, place="country/KEN", role="destination"),
        },
    )
    failures = asyncio.run(
        check_example(
            example,
            places=("country/KEN", "country/GEO"),
            names={"country/KEN": "Kenya", "country/GEO": "Georgia"},
            resolve=engine,
        )
    )
    assert [(failure.place, failure.reason) for failure in failures] == [
        ("country/GEO", "scope_mismatch")
    ]
    assert ROLE_KEYS == {None: "observationAbout", "donor": "origin", "recipient": "destination"}


def test_the_points_check_catches_an_empty_preview():
    assert _judge(definite("Count_Person", points=0))[0] == "preview_empty"


_FAILED_READS: tuple[ObservationsOmissionReason, ...] = (
    "observations_timed_out",
    "observations_bulk_fetch_failed",
    "observations_probe_failed",
    "observations_clients_unavailable",
)
_EMPTY_PREVIEWS: tuple[ObservationsOmissionReason, ...] = (
    "observations_no_points",
    "observations_nothing_to_probe",
    "observations_no_matching_source",
    "observations_no_dispatchable_facet",
    "observations_windowed_scope_unsupported",
    "observations_converted_scope_unsupported",
    "observations_preview_too_large",
)


def test_every_omission_reason_is_a_failed_read_or_an_empty_preview():
    # A reason added to the vocabulary fails here until someone puts it on a side.
    assert sorted(_FAILED_READS + _EMPTY_PREVIEWS) == sorted(get_args(ObservationsOmissionReason))
    assert set(_EMPTY_PREVIEWS) == EMPTY_PREVIEW_REASONS


@pytest.mark.parametrize("reason", _FAILED_READS)
def test_a_preview_whose_read_failed_fails_in_the_engine(reason):
    assert _judge(omitted("Count_Person", reason))[0] == "preview_failed"
    assert "preview_failed" in INFRA_FAILURES


@pytest.mark.parametrize("reason", _EMPTY_PREVIEWS)
def test_a_preview_with_nothing_to_show_fails_the_example(reason):
    assert _judge(omitted("Count_Person", reason))[0] == "preview_empty"
    assert "preview_empty" not in INFRA_FAILURES


@pytest.mark.parametrize(
    ("response", "reason"),
    [(no_data(), "no_definite_part"), (catalogue_answer(), "not_a_resolution")],
)
def test_other_answers_fail(response, reason):
    assert _judge(response)[0] == reason


def test_an_answer_split_into_several_parts_fails_even_when_one_holds_the_pin():
    [pinned] = definite("Count_Person").model_dump(mode="json")["parts"]
    [missing] = no_data().model_dump(mode="json")["parts"]
    response = serving(
        RESOLVE_RESPONSE_ADAPTER.validate_python(base_response(parts=[pinned, missing])),
        "Count_Person",
    )
    assert _judge(response) == ("multiple_parts", ("Count_Person",))


def test_test_places_run_only_where_the_stat_var_has_data():
    assert check_places(_EXAMPLE, frozenset({"country/KEN", "country/GEO"})) == (
        "country/KEN",
        "country/GEO",
    )
    assert check_places(_EXAMPLE, frozenset({"country/KEN", "country/TUV", "country/GEO"})) == (
        "country/KEN",
        "country/TUV",
        "country/GEO",
    )
    tuvalu = Example(id=0, template="x {place}", stat_var="v", place="country/TUV")
    assert check_places(tuvalu, frozenset({"country/TUV"})) == ("country/TUV",)


def test_check_example_names_each_failed_run():
    engine = FakeEngine(
        elected={"Population of Kenya": Elected("Count_Person", "country/KEN")},
        answers={"Population of Georgia": definite("Count_Person", place="country/GEO", points=0)},
    )
    failures = asyncio.run(
        check_example(
            _EXAMPLE,
            places=("country/KEN", "country/GEO", "country/TUV"),
            names={"country/KEN": "Kenya", "country/GEO": "Georgia"},
            resolve=engine,
        )
    )
    assert failures == [
        FailedRun(
            place="country/GEO",
            text="Population of Georgia",
            reason="preview_empty",
            stat_vars=("Count_Person",),
        ),
        FailedRun(place="country/TUV", text=None, reason="place_unnamed"),
    ]
    assert sorted(engine.queries) == ["Population of Georgia", "Population of Kenya"]


def _check_kenya(engine: FakeEngine) -> list[FailedRun]:
    return asyncio.run(
        check_example(
            _EXAMPLE, places=("country/KEN",), names={"country/KEN": "Kenya"}, resolve=engine
        )
    )


def test_an_engine_error_on_both_tries_fails_the_run():
    engine = FakeEngine(elected={}, answers={"Population of Kenya": RuntimeError("upstream 503")})
    [failure] = _check_kenya(engine)
    assert failure.reason == "engine_error"
    assert failure.detail is not None
    assert "upstream 503" in failure.detail
    assert engine.queries == ["Population of Kenya"] * 2


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(RuntimeError("upstream 429"), id="engine-error"),
        pytest.param(timed_out(no_data()), id="timed-out"),
        pytest.param(omitted("Count_Person", "observations_probe_failed"), id="preview-failed"),
    ],
)
def test_a_run_the_engine_fails_once_passes_on_its_retry(first):
    engine = FakeEngine(
        elected={"Population of Kenya": Elected("Count_Person", "country/KEN")},
        first={"Population of Kenya": first},
    )
    assert _check_kenya(engine) == []
    assert engine.queries == ["Population of Kenya"] * 2


def test_a_timed_out_answer_on_both_tries_fails_the_run_as_timed_out():
    engine = FakeEngine(
        elected={}, answers={"Population of Kenya": timed_out(definite("Count_Person"))}
    )
    [failure] = _check_kenya(engine)
    assert failure == FailedRun(place="country/KEN", text="Population of Kenya", reason="timed_out")


def test_a_wrong_answer_is_not_retried():
    engine = FakeEngine(
        elected={}, answers={"Population of Kenya": definite("Count_Person_Female")}
    )
    [failure] = _check_kenya(engine)
    assert failure.reason == "stat_var_not_elected"
    assert engine.queries == ["Population of Kenya"]


def test_a_preview_whose_read_fails_on_both_tries_fails_the_run_as_preview_failed():
    engine = FakeEngine(
        elected={},
        answers={"Population of Kenya": omitted("Count_Person", "observations_bulk_fetch_failed")},
    )
    [failure] = _check_kenya(engine)
    assert failure.reason == "preview_failed"
    assert engine.queries == ["Population of Kenya"] * 2


def test_a_preview_with_no_points_is_not_retried():
    engine = FakeEngine(
        elected={},
        answers={"Population of Kenya": omitted("Count_Person", "observations_no_points")},
    )
    [failure] = _check_kenya(engine)
    assert failure.reason == "preview_empty"
    assert engine.queries == ["Population of Kenya"]


def test_an_input_refusal_fails_the_example_without_a_retry():
    engine = FakeEngine(
        elected={},
        answers={"Population of Kenya": EngineInputError("query refused", code="request_invalid")},
    )
    [failure] = _check_kenya(engine)
    assert failure == FailedRun(
        place="country/KEN",
        text="Population of Kenya",
        reason="engine_refused",
        detail="request_invalid",
    )
    assert failure.reason not in INFRA_FAILURES
    assert engine.queries == ["Population of Kenya"]
