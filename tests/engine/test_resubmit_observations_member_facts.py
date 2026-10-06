"""Issue #881: a resubmit passes declared role facts to `_build_observations`.
The creditor/debtor fixture checks that both roles reach the builder.
"""

from __future__ import annotations

from unittest.mock import patch

from search.engine.region import promote_finalize, resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import ResubmitIdentityPayload
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine.test_resubmit_donor_role_e2e import (
    _FLOW_ROLE_NAMES,
    _FRA,
    _KEN,
    _SV_DCID,
    _bilateral_arcs,
    _coverage_rows_for,
    _facet,
    _fresh_discover,
    _nodes,
)
from tests.fixtures import FakeGraph


def test_resubmit_promote_threads_member_facts_into_build_observations(monkeypatch):
    """Both roles bound (mirrors test_resubmit_donor_role_e2e.py's own case
    (b), no discovery): `include_observations=True` reaches
    `_phase_finalize`'s `_build_observations` call, patched here to capture
    its kwargs. `member_facts` must no longer be the omitted default -- it
    must carry an `ArcsMemberFacts`-backed entry for `_SV_DCID` whose
    `read_declared_roles()` reports the family's real two-role signature,
    built from `state.resolved_arcs` at zero extra graph reads."""
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_rows_for([_KEN], [_SV_DCID], origin_dcids=[_FRA]),
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh_sdmx = FakeSdmxClient(observations=observations, availability={})
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=fresh_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    resubmit_sdmx = FakeSdmxClient(observations=observations, availability={})
    graph = FakeGraph(
        nodes=_nodes(
            resolve=resolve, label_dcids=["source/A"], sv_dcid=_SV_DCID, sv_arcs=_bilateral_arcs()
        ),
        detect={},
        resolve={},
        obs={},
    )
    fresh_spec = fresh.specs[0]
    identity_entities = dict(fresh_spec.definition.identity.entities)
    identity_entities["origin"] = [_FRA]
    identity_entities["destination"] = [_KEN]
    inp = ResubmitIdentityPayload(
        identity=fresh_spec.definition.identity.model_copy(update={"entities": identity_entities}),
    )

    captured_kwargs: list[dict] = []

    def _spy(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return None, None

    with patch.object(promote_finalize, "_build_observations", _spy):
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=resubmit_sdmx,
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=walk_options(graph, EngineOptions(include_observations=True)),
        )

    assert region.status == "definite", region.no_data_reason
    assert len(captured_kwargs) == 1
    member_facts = captured_kwargs[0]["member_facts"]
    assert member_facts is not None
    assert member_facts[_SV_DCID].read_declared_roles() == ("creditor", "debtor")
