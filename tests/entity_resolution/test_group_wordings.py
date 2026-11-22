"""The wording aliases generated from each group's own names."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from resolvekit import ResolutionStatus

from entity_resolution import EntityResolver, Group, group_wordings
from entity_resolution.group_wordings import (
    QueryAlias,
    Skipped,
    Unnamed,
    Wording,
    build_group_wordings,
    normalize_key,
)

from .conftest import FakeGraphLookup


def test_each_wording_resolves_to_the_group_it_was_generated_for(shared_resolver):
    """Resolving every generated text through the resolver gives the group it was built for,
    whatever other entity shares a word of its name."""
    aliases = shared_resolver.group_wordings.aliases
    assert aliases
    resolver = EntityResolver(graph=FakeGraphLookup(nodes={}), shared=shared_resolver)
    texts = sorted(aliases)

    results = asyncio.run(resolver.resolve(texts))

    wrong = {
        text: result
        for text, result in zip(texts, results, strict=True)
        if not isinstance(result, Group)
        or result.dcid != shared_resolver.groups.by_entity_id(aliases[text].group).ref
    }
    assert not wrong


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("G7 countries", "G7"),
        ("G-7 countries", "G-7"),
        ("members of the G7", "G7"),
        ("countries in the G7", "G7"),
        ("all G20 countries", "G20"),
        ("UN members", "UN"),
        ("OECD DAC donors", "OECD DAC"),
        ("DAC donor countries", "DAC"),
        ("ASEAN member states", "ASEAN"),
    ],
)
def test_a_common_group_wording_is_generated(shared_resolver, text: str, name: str):
    assert shared_resolver.group_wordings.aliases[normalize_key(text)].name == name


def test_curated_aliases_and_listed_names_are_not_generated(shared_resolver):
    """A curated row keeps its key ("DAC countries" reads as the DAC itself), a text a record
    lists resolves whole, and "AU" names Australia beside the African Union, so no wording
    holds it."""
    wordings = shared_resolver.group_wordings

    assert {"dac countries", "g20 countries", "non-dac countries"}.isdisjoint(wordings.aliases)
    assert Unnamed(group="groups/AU", name="AU", reason="gate") in wordings.unnamed
    assert "au countries" not in wordings.aliases
    assert "african union countries" in wordings.aliases


def _registry(*entity_ids: str) -> Any:
    return cast(Any, [SimpleNamespace(entity_id=entity_id) for entity_id in entity_ids])


def _patch(monkeypatch, *, listed, names, wordings=("{} countries", "{} members")):
    monkeypatch.setattr(group_wordings, "WORDINGS", wordings)
    monkeypatch.setattr(
        group_wordings, "_listed_ids", lambda resolver, index, text, context: listed.get(text, [])
    )
    monkeypatch.setattr(group_wordings, "_names", lambda resolver, entity_id: names[entity_id])


def test_a_text_two_groups_or_another_entity_name_is_left_out(monkeypatch):
    """Group A lists "Shared" and group B "SHARED", so both generate "shared countries";
    "b members" is another entity's name; "a countries" is curated; B's name "Taken" lists
    another entity first, so no wording holds it; "a members" is listed by A alone and A reads as itself, so it
    is recorded as self-listed."""
    listed = {
        "A": ["groups/A"],
        "Shared": ["groups/A"],
        "B": ["groups/B"],
        "SHARED": ["groups/B"],
        "Taken": ["country/X", "groups/B"],
        "b members": ["org/BMembers"],
        "a members": ["groups/A"],
    }
    names = {"groups/A": ["A", "Shared"], "groups/B": ["B", "SHARED", "Taken"]}
    _patch(monkeypatch, listed=listed, names=names)

    result = build_group_wordings(
        cast(Any, None),
        cast(Any, None),
        _registry("groups/A", "groups/B"),
        curated={"a countries": QueryAlias(name="A")},
    )

    assert result.aliases == {"b countries": Wording(name="B", group="groups/B", members=False)}
    assert result.unnamed == (Unnamed(group="groups/B", name="Taken", reason="gate"),)
    assert set(result.skipped) == {
        Skipped(text="a countries", group="groups/A", reason="curated"),
        Skipped(text="b members", group="groups/B", reason="listed", owners=("org/BMembers",)),
        Skipped(text="a members", group="groups/A", reason="self_listed"),
        *(
            Skipped(text=text, group=group, reason="shared", owners=("groups/A", "groups/B"))
            for text in ("shared countries", "shared members")
            for group in ("groups/A", "groups/B")
        ),
    }


def test_a_name_that_already_holds_a_set_noun_or_qualifier_is_not_worded(monkeypatch):
    """A name ending in a set noun, a ", Total" suffix or a bracketed qualifier would build
    "DAC countries countries" and the like, so no wording holds it."""
    names = ["Africa (DAC)", "DAC countries", "Donors, Total", "Plain"]
    listed = {name: ["groups/A"] for name in names}
    _patch(monkeypatch, listed=listed, names={"groups/A": names}, wordings=("{} countries",))

    result = build_group_wordings(
        cast(Any, None), cast(Any, None), _registry("groups/A"), curated={}
    )

    assert set(result.aliases) == {"plain countries"}
    assert result.unnamed == tuple(
        Unnamed(group="groups/A", name=name, reason=reason)
        for name, reason in zip(names[:3], ("qualifier", "set_noun", "qualifier"), strict=True)
    )


def test_a_wording_takes_the_reading_of_the_groups_curated_row(monkeypatch):
    """ "G countries" is curated as the members of G, so "G member countries" and "G donors"
    read as members; "H members" is curated for another group's name, so it gives H no reading
    and "H countries", curated as the group itself, decides."""
    listed = {
        "G": ["groups/G"],
        "Group of G": ["groups/G"],
        "H": ["groups/H"],
        "Other": ["groups/X"],
    }
    names = {"groups/G": ["G"], "groups/H": ["H"]}
    curated = {
        "g countries": QueryAlias(name="Group of G", members=True),
        "h members": QueryAlias(name="Other", members=True),
        "h countries": QueryAlias(name="H"),
    }
    _patch(monkeypatch, listed=listed, names=names, wordings=("{} member countries", "{} donors"))

    result = build_group_wordings(
        cast(Any, None), cast(Any, None), _registry("groups/G", "groups/H"), curated=curated
    )

    assert {text: wording.members for text, wording in result.aliases.items()} == {
        "g member countries": True,
        "g donors": True,
        "h member countries": False,
        "h donors": False,
    }


def test_a_resolvekit_error_on_a_listed_check_raises(monkeypatch):
    class _Resolver:
        def resolve(self, *args, **kwargs):
            return SimpleNamespace(status=ResolutionStatus.ERROR, candidates=[])

    with pytest.raises(group_wordings.EntityResolutionError):
        group_wordings._listed_ids(cast(Any, _Resolver()), cast(Any, None), "X", cast(Any, None))


def test_a_groups_wordings_share_one_reading_across_its_names(monkeypatch):
    """ "EU countries" is curated as members under the name "EU", so every wording of the other
    name "European Union", member wording or not, reads as members too."""
    listed = {"EU": ["groups/E"], "European Union": ["groups/E"], "Union": ["groups/E"]}
    names = {"groups/E": ["EU", "European Union"]}
    curated = {"eu countries": QueryAlias(name="Union", members=True)}
    _patch(monkeypatch, listed=listed, names=names, wordings=("{} countries", "countries in {}"))

    result = build_group_wordings(
        cast(Any, None), cast(Any, None), _registry("groups/E"), curated=curated
    )

    assert {text: wording.members for text, wording in result.aliases.items()} == {
        "european union countries": True,
        "countries in eu": True,
        "countries in european union": True,
    }


def test_a_group_whose_curated_rows_disagree_reads_as_itself(monkeypatch):
    listed = {"D": ["groups/D"], "Group D": ["groups/D"]}
    curated = {
        "d countries": QueryAlias(name="Group D"),
        "d members": QueryAlias(name="Group D", members=True),
    }
    _patch(monkeypatch, listed=listed, names={"groups/D": ["D"]}, wordings=("{} economies",))

    result = build_group_wordings(
        cast(Any, None), cast(Any, None), _registry("groups/D"), curated=curated
    )

    assert result.conflicts == {"groups/D": ("d countries", "d members")}
    assert [wording.members for wording in result.aliases.values()] == [False]


def test_a_self_listed_text_stays_an_alias_only_when_the_group_reads_members(monkeypatch):
    listed = {
        "M": ["groups/M"],
        "N": ["groups/N"],
        "m economies": ["groups/M"],
        "n economies": ["groups/N"],
    }
    names = {"groups/M": ["M"], "groups/N": ["N"]}
    curated = {"m countries": QueryAlias(name="M", members=True)}
    _patch(monkeypatch, listed=listed, names=names, wordings=("{} economies",))

    result = build_group_wordings(
        cast(Any, None), cast(Any, None), _registry("groups/M", "groups/N"), curated=curated
    )

    assert set(result.aliases) == {"m economies"}
    assert result.skipped == (Skipped(text="n economies", group="groups/N", reason="self_listed"),)


def _erroring_on(bad: str, *, name: str):
    def listed_ids(resolver, index, text, context):
        if text == bad:
            raise group_wordings.EntityResolutionError(text)
        return ["groups/A"] if text in {name, "Row"} else []

    return listed_ids


def test_a_resolvekit_error_on_a_generated_text_is_skipped(monkeypatch, caplog):
    _patch(
        monkeypatch, listed={}, names={"groups/A": ["odd"]}, wordings=("{} countries", "{} members")
    )
    monkeypatch.setattr(group_wordings, "_listed_ids", _erroring_on("odd countries", name="odd"))

    result = build_group_wordings(
        cast(Any, None), cast(Any, None), _registry("groups/A"), curated={}
    )

    assert set(result.aliases) == {"odd members"}
    assert result.skipped == (Skipped(text="odd countries", group="groups/A", reason="error"),)
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_a_resolvekit_error_on_a_curated_row_raises(monkeypatch):
    _patch(monkeypatch, listed={}, names={"groups/A": ["odd"]}, wordings=("{} members",))
    monkeypatch.setattr(group_wordings, "_listed_ids", _erroring_on("Row", name="odd"))

    with pytest.raises(group_wordings.EntityResolutionError):
        build_group_wordings(
            cast(Any, None),
            cast(Any, None),
            _registry("groups/A"),
            curated={"odd countries": QueryAlias(name="Row")},
        )
