"""The wording aliases generated from each group's own names."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from entity_resolution import group_wordings
from entity_resolution.group_wordings import Skipped, build_group_wordings, normalize_key


def _name_owner(shared, aliases):
    """The group each alias's name resolves to alone, read on the resolver thread."""
    from resolvekit import ResolutionContext

    from entity_resolution.types import ADMITTED_TYPES

    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    return {
        name: group_wordings._listed_ids(shared.resolver, shared.alias_index, name, context)[0]
        for name in set(aliases.values())
    }


def test_each_wording_resolves_to_the_group_whose_name_it_holds(shared_resolver):
    """Every generated text holds the name it maps to, and that name resolves to one of the
    registry's groups first."""
    aliases = shared_resolver.group_wordings.aliases
    owners = shared_resolver.run(_name_owner, shared_resolver, aliases).result()
    groups = {group.entity_id for group in shared_resolver.groups}

    assert aliases
    for text, name in aliases.items():
        assert normalize_key(name) in text, (text, name)
        assert owners[name] in groups, (text, name, owners[name])


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
    assert shared_resolver.group_wordings.aliases[normalize_key(text)] == name


def test_curated_aliases_and_listed_names_are_not_generated(shared_resolver):
    """A curated row keeps its key ("DAC countries" reads as the DAC itself), a text a record
    lists resolves whole, and "AU" names Australia beside the African Union, so no wording
    holds it."""
    wordings = shared_resolver.group_wordings

    assert {"dac countries", "g20 countries", "non-dac countries"}.isdisjoint(wordings.aliases)
    assert ("groups/AU", "AU") in wordings.unnamed
    assert "au countries" not in wordings.aliases
    assert "african union countries" in wordings.aliases


def test_a_text_two_groups_or_another_entity_name_is_left_out(monkeypatch):
    """Group A lists "Shared" and group B "SHARED", so both generate "shared countries";
    "b members" is another entity's name; "a countries" is curated; B's name "Taken" lists
    another entity first, so no wording holds it."""
    listed = {
        "A": ["groups/A"],
        "Shared": ["groups/A"],
        "B": ["groups/B"],
        "SHARED": ["groups/B"],
        "Taken": ["country/X", "groups/B"],
        "b members": ["org/BMembers"],
    }
    names = {"groups/A": ["A", "Shared"], "groups/B": ["B", "SHARED", "Taken"]}
    monkeypatch.setattr(group_wordings, "WORDINGS", ("{} countries", "{} members"))
    monkeypatch.setattr(
        group_wordings, "_listed_ids", lambda resolver, index, text, context: listed.get(text, [])
    )
    monkeypatch.setattr(group_wordings, "_names", lambda resolver, entity_id: names[entity_id])
    registry = [SimpleNamespace(entity_id="groups/A"), SimpleNamespace(entity_id="groups/B")]

    result = build_group_wordings(
        cast(Any, None), cast(Any, None), cast(Any, registry), curated=frozenset({"a countries"})
    )

    assert result.aliases == {"a members": "A", "b countries": "B"}
    assert result.unnamed == (("groups/B", "Taken"),)
    assert set(result.skipped) == {
        Skipped(text="a countries", group="groups/A", reason="curated"),
        Skipped(text="b members", group="groups/B", reason="listed", owners=("org/BMembers",)),
        *(
            Skipped(text=text, group=group, reason="shared", owners=("groups/A", "groups/B"))
            for text in ("shared countries", "shared members")
            for group in ("groups/A", "groups/B")
        ),
    }
