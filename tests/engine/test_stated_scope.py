"""The scope words a text states, and which text each part of a request is read from.

The phrase cases are user messages from recorded Studio conversations, verbatim, beside a
few written to pin a rule the recordings do not exercise."""

from __future__ import annotations

import pytest

from code_index.declarations import DATASETS
from search.engine.stated_scope import (
    part_scope_intakes,
    read_stated_scope,
    without_scope_phrases,
)
from search.models import ChoiceOrigin, Entry, TextOrigin

_NET = "measure_basis:net"
_GROSS = "measure_basis:gross"
_GRANT_EQUIVALENT = "measure_basis:grant_equivalent"
_COMMITMENTS = "measure_basis:commitments"
_BILATERAL = "provider:bilateral"
_INCLUSIVE = "provider:inclusive"


@pytest.mark.parametrize(
    ("text", "stated", "total"),
    [
        # Recorded messages.
        (
            "Net bilateral ODA for education to Kenya",
            [(_NET, "Net"), (_BILATERAL, "bilateral")],
            False,
        ),
        (
            "What was Japan\u2019s grant-equivalent ODA in 2016?",
            [(_GRANT_EQUIVALENT, "grant-equivalent")],
            False,
        ),
        (
            "How much did the DRC receive in external aid (in grant equivalents) in 2024?",
            [(_GRANT_EQUIVALENT, "grant equivalents")],
            False,
        ),
        ("Can you show ODA commitments from 2022-2026?", [(_COMMITMENTS, "commitments")], False),
        (
            "What was the total gross ODA from all providers to Kenya in 2022?",
            [(_GROSS, "gross")],
            True,
        ),
        (
            "Let's just look at national donors, and not multilateral instutitons like GAVI and "
            "GF, who get money from these national donors anyway",
            [(_BILATERAL, "not multilateral")],
            False,
        ),
        (
            "Rank recipients of France's bilateral aid by net disbursements in 2024, including "
            "those with negative net flows.",
            [(_NET, "net"), (_BILATERAL, "bilateral")],
            False,
        ),
        ("Total health ODA to the DRC in 2024", [], True),
        ("not per capita but total", [], True),
        # A negated basis word states nothing.
        ("Show it as total UK ODA, not just gross disbursements", [], True),
        # Two bases asked for together state neither.
        ("ODA inflow to ethiopia during 2024, both gross and net ODA", [], False),
        (
            "How much official finance did the United Kingdom provide to Kenya through private "
            "sector instruments each year from 2018 to 2024, in current US dollars? Name the "
            "source dataset and show commitments and gross disbursements separately.",
            [],
            False,
        ),
        # "multilateral" naming a kind of provider beside "bilateral" leaves the axis unstated.
        (
            "Give me a breakdown of UK's ODA budget by type (bilateral/multilateral), recipient "
            "(income group, region) and sector. Data for latest available  year.",
            [],
            False,
        ),
        (
            "make a simple bar chart that shows African borrowing from domestic, bilateral, and "
            "multilateral creditors from 2000 to 2025. In that chart, separate loans from China.",
            [],
            False,
        ),
        ("France's imputed multilateral ODA to Nigeria for health in 2024.", [], False),
        # "net" as a verb, or heading another measure.
        ("Did you net the two together rather than showing two separate lines?", [], False),
        ("what is the net debt of kenya", [], False),
        # The singular "commitment" is a pledge, not a basis.
        ("Has the US kept its commitment on health aid", [], False),
        # Written cases.
        (
            "health ODA to Kenya including multilateral",
            [(_INCLUSIVE, "including multilateral")],
            False,
        ),
        (
            "bilateral plus imputed multilateral aid from the US",
            [(_INCLUSIVE, "bilateral plus imputed multilateral")],
            False,
        ),
        (
            "aid from France not including multilateral",
            [(_BILATERAL, "not including multilateral")],
            False,
        ),
        (
            "aid to Kenya excluding multilateral flows",
            [(_BILATERAL, "excluding multilateral")],
            False,
        ),
        ("non-bilateral ODA to Kenya", [], False),
        # "multilateral" heading a kind of provider after "including" or "plus" states no
        # scope.
        ("ODA to Kenya including multilateral development banks", [], False),
        ("aid to Ghana plus multilateral lenders", [], False),
        ("aid incl. the multilateral agencies", [], False),
        # Negated, it excludes that kind of provider, as "not multilateral institutions" does.
        (
            "aid from France not including multilateral institutions",
            [(_BILATERAL, "not including multilateral")],
            False,
        ),
        ("ODA to Kenya including multilaterals", [(_INCLUSIVE, "including multilaterals")], False),
        ("ODA as a share of gross national income", [], False),
        ("ODA net of debt relief", [], False),
    ],
)
def test_read_stated_scope(text, stated, total):
    scope = read_stated_scope(text)

    read = []
    for entry in scope.entries:
        origin = entry.origin
        assert isinstance(origin, TextOrigin)
        assert entry.provenance == "explicit"
        assert text[origin.start : origin.end] == origin.text
        read.append((entry.option_id, origin.text))
    assert read == stated
    assert scope.total is total


@pytest.mark.parametrize(
    ("text", "rest"),
    [
        ("France's health ODA including imputed multilateral", "France's health ODA"),
        ("ODA to Kenya, bilateral plus imputed multilateral, in 2020", "ODA to Kenya, , in 2020"),
        ("aid from France not including multilateral", "aid from France"),
        ("including imputed multilateral", ""),
        # Scope words that also name measures and parties stay, and a text without a scope
        # phrase stays as it is.
        ("net bilateral  ODA", "net bilateral  ODA"),
        # Words stating two provider options state no entry, so the phrase stays for the
        # binders and the dropped-qualifier warning.
        (
            "France's bilateral aid including imputed multilateral",
            "France's bilateral aid including imputed multilateral",
        ),
        # A phrase naming a kind of provider states no scope and stays whole.
        (
            "ODA to Kenya including multilateral development banks",
            "ODA to Kenya including multilateral development banks",
        ),
    ],
)
def test_a_scope_phrase_leaves_the_text_with_any_negation_of_it(text, rest):
    (intake,) = part_scope_intakes(message=text, part_texts=[text])

    assert without_scope_phrases(text, intake) == rest


_PHRASED = "France's health ODA including imputed multilateral"


@pytest.mark.parametrize(
    ("intake_text", "text", "rest"),
    [
        pytest.param(None, _PHRASED, _PHRASED, id="no-intake"),
        pytest.param("France's health ODA", _PHRASED, _PHRASED, id="no-entry"),
        # Only the entry a phrase reads licenses it: a bilateral entry leaves an inclusive
        # phrase and an inclusive entry a negated one.
        pytest.param("France's bilateral health ODA", _PHRASED, _PHRASED, id="bilateral-entry"),
        pytest.param(
            "health ODA including multilateral",
            "health ODA not including multilateral",
            "health ODA not including multilateral",
            id="inclusive-entry-negated-phrase",
        ),
        pytest.param(
            "health ODA excluding multilateral",
            "health ODA not including multilateral",
            "health ODA",
            id="bilateral-entry-negated-phrase",
        ),
    ],
)
def test_a_scope_phrase_goes_only_where_the_part_states_the_entry_it_reads(intake_text, text, rest):
    intake = (
        None
        if intake_text is None
        else part_scope_intakes(message=None, part_texts=[intake_text])[0]
    )

    assert without_scope_phrases(text, intake) == rest


def test_a_carried_provider_entry_reads_no_phrase():
    """A carried provider entry, or a switch, is no reading of the part's words, so the
    phrase stays."""
    text = "France's health ODA including imputed multilateral"
    (carried,) = part_scope_intakes(message=None, part_texts=["health ODA"], carried=[_CARRIED])

    assert without_scope_phrases(text, carried) == text


def test_the_provider_word_is_one_the_crs_states_of_its_rows():
    """The word "bilateral" reads as the provider scope because the CRS states it of every row
    (`DatasetDeclaration.stated_words`)."""
    crs = next(d for d in DATASETS if d.label.endswith("(CRS)"))
    assert "bilateral" in {stated.word for stated in crs.stated_words}


_CARRIED = Entry(
    axis="provider",
    option_id=_INCLUSIVE,
    provenance="carried",
    origin=ChoiceOrigin(choice_id="spec:provider"),
)


def test_one_part_reads_the_message_and_takes_every_carried_entry():
    (intake,) = part_scope_intakes(
        message="net health ODA to Kenya", part_texts=["gross health ODA"], carried=[_CARRIED]
    )

    assert [e.option_id for e in intake.entries] == [_NET, _INCLUSIVE]


def test_one_part_without_a_message_reads_its_own_text():
    (intake,) = part_scope_intakes(message=None, part_texts=["total gross health ODA"])

    assert [e.option_id for e in intake.entries] == [_GROSS]
    assert intake.total


def test_several_parts_read_their_own_text_and_never_the_message():
    """One part's words never reach another, and the message's reach none."""
    intakes = part_scope_intakes(
        message="total net bilateral aid to Kenya and GDP",
        part_texts=["net aid to Kenya", "GDP of Kenya"],
        carried=[_CARRIED],
    )

    assert [[e.option_id for e in i.entries] for i in intakes] == [
        [_NET, _INCLUSIVE],
        [_INCLUSIVE],
    ]
    assert [i.total for i in intakes] == [False, False]
