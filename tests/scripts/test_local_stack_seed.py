"""The local stack's seed readings mint, and serve their fixture points, over the upstream
doubles `scripts/local-stack/upstream.py` builds.

The local stack runs outside CI, so this is where a mint change that the committed fixture no
longer satisfies (a graph read the doubles do not answer, a scope the door refuses) fails.
Each reading is minted through `POST /api/datasets` exactly as `seed.py` posts it, over
in-memory stores, and its observations must equal the fixture's points.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from artifact_store import LiveArtifactInventory
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.version_store import LiveVersionRecordWriter
from materialise.rate_tables import RateTableCache
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.datasets_api._app_factories import make_client
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.datasets_api.conftest import FakeDatasetStore
from tests.datasets_api.test_pin_door import FakeArtifactReferenceWriter
from tests.datasets_api.test_snapshot_lifecycle import _InventoryRecordingArtifactStore
from tests.origin_catalogue.fake_catalogue import fake_catalogue_service

_LOCAL_STACK = Path(__file__).parents[2] / "scripts" / "local-stack"


def _load(name: str, *, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _LOCAL_STACK / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# seed.py imports its sibling as `upstream`, as it does when run from its own directory.
upstream = _load("upstream", module_name="upstream")
seed = _load("seed", module_name="local_stack_seed")

_FIXTURE = upstream.load_fixture()


@pytest.fixture
def client():
    version_store = FakeVersionRecordDatasetStore()
    spec_store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=spec_store),
        spec_reader=LiveSpecReader(store=spec_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        version_writer=LiveVersionRecordWriter(store=version_store),
        artifact_store=_InventoryRecordingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        graph=upstream.graph_reader(_FIXTURE),
        sdmx=upstream.sdmx_client(_FIXTURE),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=upstream.coverage(_FIXTURE),
        rate_cache=RateTableCache(),
        catalogue=fake_catalogue_service(),
    )


@pytest.mark.parametrize("reading", _FIXTURE.readings, ids=lambda reading: reading.name)
def test_each_seed_reading_mints_and_serves_its_fixture_points(client, reading):
    (seeded,) = seed.seed(client, [reading])

    served = client.get(
        f"/api/datasets/{seeded['spec_id']}/{seeded['content_version']}/observations"
    )

    assert served.status_code == 200, served.text
    assert {
        tuple(sorted(entity["ref"] for entity in series["entities"])): [
            (point["date"], point["value"]) for point in series["points"]
        ]
        for series in served.json()["series"]
    } == {tuple(sorted(series.entities.values())): list(series.points) for series in reading.series}
