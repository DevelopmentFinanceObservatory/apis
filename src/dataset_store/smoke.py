"""Check the runtime role's Postgres login with `SELECT 1`.

The deployed image omits `scripts/`, so this command lives in the package.
Run it with `python -m dataset_store.smoke`.
"""

from __future__ import annotations

import sys

from dataset_store.connection import ConnectionSettings
from dataset_store.errors import DatasetStoreConfigError
from dataset_store.provider import close_shared_pool, shared_pool
from dataset_store.store import LiveDatasetStore


def main() -> int:
    try:
        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_RUNTIME_DB_USER",
            optional_password_var="DATASET_STORE_RUNTIME_DB_PASSWORD",
            enable_iam_auth=True,
        )
    except DatasetStoreConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    store = LiveDatasetStore(connections=shared_pool())
    try:
        with store.transaction() as tx:
            row = tx.fetch_one("SELECT 1 AS one")
        assert row == {"one": 1}, row
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        close_shared_pool()
    print(f"ok: connected as {settings.user}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
