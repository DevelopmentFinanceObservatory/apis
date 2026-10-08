"""Tests admission of expanded topic members before the member cap."""

from __future__ import annotations

from origin_catalogue import BASE, UN_SYSTEM
from search.engine.config import settings
from search.engine.retrieval.topics import expand_topics
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_BASE_PROVENANCE = "dc/base/WHO_GHO"
_UN_PROVENANCE = "undata/p/WHO"


def _topic_graph(members: dict[str, list[str]]) -> FakeGraph:
    nodes = {
        topic: {"arcs": {"relevantVariable": {"nodes": [{"dcid": leaf} for leaf in leaves]}}}
        for topic, leaves in members.items()
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _catalogue(
    *,
    base_variables: tuple[str, ...] = (),
    withheld: tuple[str, ...] = (),
    successors: dict[str, str | None] | None = None,
):
    """Admits `base_variables` and refuses `withheld`, the variables whose data only
    unadmitted provenances hold; `successors` names the superseded ones among them. The UN
    tables hold `undata/who/U5MR` and exclude `undata/who/ODA`."""
    return restrictive_catalogue(
        provenances=[
            provenance_row(_BASE_PROVENANCE, BASE, source="dc/s/WorldHealthOrganization"),
            provenance_row(_UN_PROVENANCE, UN_SYSTEM, segment="who"),
        ],
        base_variables={variable: ([_BASE_PROVENANCE], []) for variable in base_variables},
        withheld_variables=withheld,
        successors=successors,
        un_mirror=un_mirror(
            {"undata/who/U5MR": un_variable("undata/who/SERIES-U5MR", name="Under-five deaths")},
            excluded={"undata/who/ODA": "undata/who/SERIES-ODA"},
        ),
    )


class TestTopicMemberAdmission:
    def test_one_admitted_member_among_many_excluded_is_kept(self, bind_catalogue):
        excluded = [f"Excluded_{i}" for i in range(settings().topic_expand_max_members + 10)]
        bind_catalogue(_catalogue(base_variables=("Admitted_Leaf",), withheld=tuple(excluded)))
        graph = _topic_graph({"dc/topic/Mixed": [*excluded, "Admitted_Leaf"]})

        assert expand_topics(["dc/topic/Mixed"], graph=graph) == {"Admitted_Leaf": "dc/topic/Mixed"}

    def test_excluded_members_do_not_count_toward_the_cap(self, bind_catalogue):
        bind_catalogue(
            _catalogue(base_variables=("Leaf_A", "Leaf_B"), withheld=("Excluded_1", "Excluded_2"))
        )
        graph = _topic_graph({"dc/topic/Small": ["Excluded_1", "Leaf_A", "Excluded_2", "Leaf_B"]})

        leaf_to_root = expand_topics(["dc/topic/Small"], graph=graph, max_members=2)

        assert set(leaf_to_root) == {"Leaf_A", "Leaf_B"}

    def test_a_topic_that_keeps_no_member_is_dropped(self, bind_catalogue):
        bind_catalogue(
            _catalogue(base_variables=("Leaf_A",), withheld=("Excluded_1", "Excluded_2"))
        )
        graph = _topic_graph(
            {"dc/topic/Kept": ["Leaf_A"], "dc/topic/Empty": ["Excluded_1", "Excluded_2"]}
        )

        leaf_to_root = expand_topics(["dc/topic/Empty", "dc/topic/Kept"], graph=graph)

        assert leaf_to_root == {"Leaf_A": "dc/topic/Kept"}

    def test_un_members_pass_when_the_un_tables_hold_them(self, bind_catalogue):
        bind_catalogue(_catalogue())
        graph = _topic_graph(
            {
                "undata/topic/who/Mortality": [
                    "undata/who/ODA",
                    "undata/who/U5MR",
                    "undata/who/NOT_IN_THE_BUILD",
                ]
            }
        )

        assert expand_topics(["undata/topic/who/Mortality"], graph=graph) == {
            "undata/who/U5MR": "undata/topic/who/Mortality"
        }

    def test_a_member_under_a_superseded_base_provenance_stands_as_its_successor_or_drops(
        self, bind_catalogue
    ):
        """`sdg/SH_DYN_MORT` is superseded with a successor and `sdg/SH_DYN_GONE` with none:
        the first stands as the UN variable, the second is dropped."""
        superseded = {"sdg/SH_DYN_MORT": "undata/who/U5MR", "sdg/SH_DYN_GONE": None}
        bind_catalogue(
            _catalogue(
                base_variables=("Leaf_A",), withheld=tuple(superseded), successors=superseded
            )
        )
        graph = _topic_graph({"dc/topic/sdg_3.2.1": [*superseded, "Leaf_A"]})

        assert expand_topics(["dc/topic/sdg_3.2.1"], graph=graph) == {
            "undata/who/U5MR": "dc/topic/sdg_3.2.1",
            "Leaf_A": "dc/topic/sdg_3.2.1",
        }
