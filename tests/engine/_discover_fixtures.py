"""Shared StatVar, SDG taxonomy, and graph fixtures for discover tests."""

from __future__ import annotations

from tests.fixtures import FakeGraph

POP_TYPE_DCID = "CreditorDebtorFlow"
MEAS_PROP_DCID = "flowAmount"
STAT_TYPE_DCID = "measuredValue"
MEAS_QUAL_DCID: str | None = None
MEAS_DENOM_DCID: str | None = None
PROP_SCHEME = "FlowScheme"
PROP_PURPOSE = "FlowPurpose"
PROP_RECIPIENT = "FlowDestination"

# A namespace prefix for self-authored test SV dcids below, distinguishing
# them from any real StatVar dcid.
_TEST_SV_NAMESPACE = "CrsShapedFixture/"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _three_constraint_arcs(purpose: str, scheme: str, recipient: str) -> dict:
    """Build a minimal node_arcs dict for an SV carrying the three-constraint
    five-tuple above."""
    return {
        "populationType": {"nodes": [{"dcid": POP_TYPE_DCID}]},
        "measuredProperty": {"nodes": [{"dcid": MEAS_PROP_DCID}]},
        "statType": {"nodes": [{"dcid": STAT_TYPE_DCID}]},
        "constraintProperties": {
            "nodes": [
                {"dcid": PROP_SCHEME},
                {"dcid": PROP_PURPOSE},
                {"dcid": PROP_RECIPIENT},
            ]
        },
        PROP_SCHEME: {"nodes": [{"dcid": scheme}]},
        PROP_PURPOSE: {"nodes": [{"dcid": purpose}]},
        PROP_RECIPIENT: {"nodes": [{"dcid": recipient}]},
    }


def _fake_graph_with_svs(sv_arcs: dict[str, dict]) -> FakeGraph:
    """Build a graph with the three constraint-property labels and supplied StatVars."""
    nodes: dict = {
        PROP_SCHEME: {"label": "Flow Scheme"},
        PROP_PURPOSE: {"label": "Flow Purpose"},
        PROP_RECIPIENT: {"label": "Flow Destination"},
    }
    for sv_dcid, arcs in sv_arcs.items():
        nodes[sv_dcid] = {"label": f"Label for {sv_dcid}", "arcs": arcs}
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _std_sv_arcs(
    pop_type: str = "Person", meas_prop: str = "count", name: str | None = None
) -> dict:
    """Build standard SV arcs, with an optional name arc."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if name is not None:
        arcs["name"] = {"nodes": [{"value": name}]}
    return arcs


def _sdg_taxonomy_graph() -> FakeGraph:
    """Model the live three-hop HIV taxonomy from Topic to peer group to sex-specific StatVars.

    Arc targets carry their `types`, as in the live API.
    """
    nodes = {
        "dc/topic/HIV": {
            "arcs": {
                "relevantVariable": {"nodes": [{"dcid": "dc/topic/sdg_3.3.1", "types": ["Topic"]}]}
            }
        },
        "dc/topic/sdg_3.3.1": {
            "arcs": {
                "relevantVariable": {
                    "nodes": [{"dcid": "dc/topic/sdgSHHIVINCD", "types": ["Topic"]}]
                }
            }
        },
        "dc/topic/sdgSHHIVINCD": {
            "arcs": {
                "relevantVariable": {
                    "nodes": [{"dcid": "dc/svpg/SDGSHHIVINCD.4", "types": ["StatVarPeerGroup"]}]
                }
            }
        },
        "dc/svpg/SDGSHHIVINCD.4": {
            "arcs": {
                "member": {
                    "nodes": [
                        {"dcid": "sdg/SH_HIV_INCD.SEX--F", "types": ["StatisticalVariable"]},
                        {"dcid": "sdg/SH_HIV_INCD.SEX--M", "types": ["StatisticalVariable"]},
                    ]
                }
            }
        },
        "sdg/SH_HIV_INCD.SEX--F": {
            "label": "female",
            "arcs": _std_sv_arcs("SDG_SH_HIV_INCD", "value"),
        },
        "sdg/SH_HIV_INCD.SEX--M": {
            "label": "male",
            "arcs": _std_sv_arcs("SDG_SH_HIV_INCD", "value"),
        },
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
