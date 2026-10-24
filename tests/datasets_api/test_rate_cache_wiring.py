"""Each door reuses one `RateTableCache` across materialisations and passes it to the
materialise entry point. A spy checks cache identity across definition calls, and that the
door passes its code index's absent-observation rule.
"""

from __future__ import annotations

import datasets_api.serve as serve_module
from materialise.rate_tables import RateTableCache
from tests.code_index.aid_identity import CRS_PROVENANCE
from tests.code_index.support import INDEX_FIXTURE, load_index

from .test_mint_spec_id import _STATVAR_SET_BODY, _spec_id_body
from .test_mint_spec_id import _client as _spec_client


def test_definition_door_shares_one_rate_cache_across_two_materialisations(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_group = serve_module.materialise_group

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", spy)

    from ._artifact_doubles import FakeArtifactStore

    client = _spec_client(artifact_store=FakeArtifactStore())

    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id))
    assert resp.status_code == 200, resp.text

    assert len(recorded) == 2
    assert recorded[0] is not None
    assert isinstance(recorded[0], RateTableCache)
    assert recorded[0] is recorded[1]


def test_the_definition_door_reads_absence_as_its_code_index_declares(monkeypatch):
    """The group's out-of-window members follow the absent-observation rule the rendered sum
    reads (`absent_member_reads_zero`), over the door's own code index, whichever path
    resolved the members."""
    recorded: list[object] = []
    real_materialise_group = serve_module.materialise_group

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("absent_reads_zero"))
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", spy)

    from ._artifact_doubles import FakeArtifactStore

    client = _spec_client(artifact_store=FakeArtifactStore(), code_index=load_index(INDEX_FIXTURE))

    resp = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert resp.status_code == 200, resp.text

    [absent_reads_zero] = recorded
    assert callable(absent_reads_zero)
    assert (absent_reads_zero(CRS_PROVENANCE), absent_reads_zero("provenance/Undeclared")) == (
        True,
        False,
    )
