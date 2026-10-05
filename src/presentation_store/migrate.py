"""Apply `presentation_store` migrations with its own settings and bookkeeping table."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dataset_store import connection as dataset_store_connection
from dataset_store.errors import DatasetStoreError
from dataset_store.migrate import apply_pending
from presentation_store.connection import PresentationStoreSettings
from presentation_store.errors import PresentationStoreConfigError

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_BOOKKEEPING_SCHEMA = "presentation"
_BOOKKEEPING_TABLE = "presentation.presentation_store_schema_migrations"


def main(argv: Sequence[str] | None = None) -> int:
    """Apply migrations. Print configuration or migration errors to stderr and return 1."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m presentation_store.migrate")
    parser.add_argument("command", choices=["up"])
    args = parser.parse_args(argv)
    assert args.command == "up"

    try:
        settings = PresentationStoreSettings.from_env(
            user_var="PRESENTATION_STORE_MIGRATOR_DB_USER",
            password_var="PRESENTATION_STORE_MIGRATOR_DB_PASSWORD",
            enable_iam_auth=False,
        )
        runtime_user = os.environ["PRESENTATION_STORE_RUNTIME_DB_USER"]
    except (PresentationStoreConfigError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    connector = (
        dataset_store_connection.build_connector() if settings.instance_connection_name else None
    )
    try:
        applied = apply_pending(
            connect=settings.connect_callable(connector=connector),
            migrations_dir=MIGRATIONS_DIR,
            substitutions={"RUNTIME_DB_USER": runtime_user},
            bookkeeping_schema=_BOOKKEEPING_SCHEMA,
            bookkeeping_table=_BOOKKEEPING_TABLE,
        )
    except (DatasetStoreError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if applied:
        print(f"applied: {', '.join(applied)}")
    else:
        print("nothing to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
