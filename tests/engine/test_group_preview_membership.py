"""Confirmed `sv_dcids` supply group previews; enumeration truncates at 4,000 candidates."""

from __future__ import annotations

import statvar_members
from search.engine.retrieval.member_facts import ArcsMemberFacts
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import build_observations_planned, materialise_double_from_obs
from tests.fixtures import FakeGraph


def _member_facts(constraints: dict[str, str]) -> ArcsMemberFacts:
    arcs = {
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }
    return ArcsMemberFacts(arcs=arcs)


def _unreachable_enumerate_members(*args: object, **kwargs: object) -> None:
    """Fail if standard previews call `enumerate_members`."""
    raise AssertionError("enumerate_members must not be called on the standard preview path")


# Standard previews pair confirmed coverage members after `_minimal_constraint_svs` filtering.


def test_standard_group_whose_family_would_truncate_enumeration_still_previews_from_coverage(
    monkeypatch,
):
    """Person/count exceeds enumeration's 4,000-candidate bound (#228).

    The fake graph has no member arcs. Enumeration raises if called, so coverage must supply both
    members."""
    monkeypatch.setattr(statvar_members, "enumerate_members", _unreachable_enumerate_members)

    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "LifeExpectancy_Person_Male|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 61.0}],
                }
            ],
            "LifeExpectancy_Person_Female|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 66.0}],
                }
            ],
        }
    )

    observations, warn = build_observations_planned(
        ["LifeExpectancy_Person_Male", "LifeExpectancy_Person_Female"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        member_facts={
            "LifeExpectancy_Person_Male": _member_facts({"gender": "Male"}),
            "LifeExpectancy_Person_Female": _member_facts({"gender": "Female"}),
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert {s.stat_var.ref for s in observations.series} == {
        "LifeExpectancy_Person_Male",
        "LifeExpectancy_Person_Female",
    }


def test_dominated_member_never_appears_beside_its_dominating_sibling(monkeypatch):
    """When a family total and part both appear, only the total previews.
    Drop the part before building pairs."""
    monkeypatch.setattr(statvar_members, "enumerate_members", _unreachable_enumerate_members)

    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Flow_Total|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 100.0}],
                }
            ],
            "Flow_BySectorHealth|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 40.0}],
                }
            ],
        }
    )

    observations, warn = build_observations_planned(
        ["Flow_Total", "Flow_BySectorHealth"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        member_facts={
            "Flow_Total": _member_facts({}),
            "Flow_BySectorHealth": _member_facts({"purpose": "Health"}),
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert {s.stat_var.ref for s in observations.series} == {"Flow_Total"}


def test_standard_group_at_or_under_ceiling_previews_exactly_its_restricted_members(monkeypatch):
    """Comfortably under `_OBS_PREVIEW_MAX_PAIRS`: Male/Female are disjoint
    (neither's own constraint-key set dominates the other's), so
    `_minimal_constraint_svs` restricts nothing here -- both preview against
    every entity, exactly the (stat_var, entity) pairs sv_dcids x entities
    denotes, no more and no fewer."""
    monkeypatch.setattr(statvar_members, "enumerate_members", _unreachable_enumerate_members)

    entities = ["country/KEN", "country/TZA"]
    svs = ["LifeExpectancy_Person_Male", "LifeExpectancy_Person_Female"]
    obs = {
        f"{sv}|{entity}": [
            {
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
                "observations": [{"date": "2020", "value": 50.0}],
            }
        ]
        for sv in svs
        for entity in entities
    }
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)

    observations, warn = build_observations_planned(
        svs,
        {"observationAbout": entities},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        member_facts={
            "LifeExpectancy_Person_Male": _member_facts({"gender": "Male"}),
            "LifeExpectancy_Person_Female": _member_facts({"gender": "Female"}),
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert len(observations.series) == len(svs) * len(entities)
    assert {s.stat_var.ref for s in observations.series} == set(svs)
