"""Re-export spec-registry fixtures and provide the shared SQLite version store.
Other datasets API test doubles live in sibling modules.
"""

import pytest

from tests.spec_registry.conftest import (  # noqa: F401 -- re-exported for datasets_api tests
    FakeDatasetStore,
    FakeRegistryState,
    FakeTransaction,
    identity_of_encoded_size,
    minimal_identity,
    wire_definition_for,
)

from ._version_record_store import FakeVersionRecordDatasetStore


@pytest.fixture
def version_store():
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()
