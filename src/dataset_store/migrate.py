"""Forward-only migrations. Fix an applied migration with a new file.

SQL files run in filename order, each in its own transaction. The bookkeeping
table stores each file's raw-byte SHA-256 and detects edits, deletion, or
renaming of applied files. Placeholder values and bookkeeping identifiers
are validated before interpolation into DDL.

Each file runs in one `cursor.execute()` call. With no parameters, pg8000
uses Postgres's simple-query protocol, which accepts multiple statements.
Splitting on semicolons would break semicolons inside string literals.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from dataset_store import connection
from dataset_store.dbapi import DBAPIConnection
from dataset_store.errors import DatasetStoreConfigError, DatasetStoreError

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_BOOKKEEPING_SCHEMA = "datasets"
_BOOKKEEPING_TABLE = "datasets.dataset_store_schema_migrations"
_VALID_SUBSTITUTION = re.compile(r"^[A-Za-z0-9@._-]+$")
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_VALID_BOOKKEEPING_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]*$")
_VALID_BOOKKEEPING_TABLE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")


class MigrationTemplateError(DatasetStoreError):
    """A placeholder is undefined or its value fails the identifier-safety check."""


def _validate_bookkeeping_identifiers(*, bookkeeping_schema: str, bookkeeping_table: str) -> None:
    """Validate DDL identifiers before SQL runs. Invalid names raise `ValueError`."""
    if not _VALID_BOOKKEEPING_SCHEMA.fullmatch(bookkeeping_schema):
        raise ValueError(
            f"invalid bookkeeping_schema {bookkeeping_schema!r}: must match "
            f"{_VALID_BOOKKEEPING_SCHEMA.pattern}"
        )
    if not _VALID_BOOKKEEPING_TABLE.fullmatch(bookkeeping_table):
        raise ValueError(
            f"invalid bookkeeping_table {bookkeeping_table!r}: must match "
            f"{_VALID_BOOKKEEPING_TABLE.pattern}"
        )


class MigrationChecksumError(DatasetStoreError):
    """An applied migration changed or disappeared. Add a new migration to fix it."""


def _content_hash(path: Path) -> str:
    """Return the raw-file SHA-256, before environment-specific substitution."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    """Every `*.sql` file directly under `directory`, sorted by filename."""
    return sorted(directory.glob("*.sql"))


def render(sql_text: str, *, substitutions: Mapping[str, str]) -> str:
    """Substitute `{{KEY}}` tokens.

    Raise `MigrationTemplateError` for missing keys or unsafe values.
    """

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in substitutions:
            raise MigrationTemplateError(
                f"migration references undefined placeholder {{{{{key}}}}}"
            )
        value = substitutions[key]
        if not _VALID_SUBSTITUTION.fullmatch(value):
            raise MigrationTemplateError(
                f"substitution value for {key} contains characters outside "
                f"[A-Za-z0-9@._-]: {value!r}"
            )
        return value

    return _PLACEHOLDER.sub(_sub, sql_text)


def _ensure_bookkeeping(
    conn: DBAPIConnection, *, bookkeeping_schema: str, bookkeeping_table: str
) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {bookkeeping_schema}")
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {bookkeeping_table} "
            "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now(), "
            "content_sha256 text)"
        )
        # Add `content_sha256` when absent, preserving existing migration rows.
        cursor.execute(
            f"ALTER TABLE {bookkeeping_table} ADD COLUMN IF NOT EXISTS content_sha256 text"
        )
    finally:
        cursor.close()
    conn.commit()


def _applied_checksums(conn: DBAPIConnection, *, bookkeeping_table: str) -> dict[str, str | None]:
    """Map recorded versions to their hash, using `None` for legacy rows."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT version, content_sha256 FROM {bookkeeping_table}")
        return {cast(str, row[0]): cast("str | None", row[1]) for row in cursor.fetchall()}
    finally:
        cursor.close()


def apply_pending(
    *,
    connect: Callable[[], DBAPIConnection],
    migrations_dir: Path = MIGRATIONS_DIR,
    substitutions: Mapping[str, str],
    bookkeeping_schema: str = _BOOKKEEPING_SCHEMA,
    bookkeeping_table: str = _BOOKKEEPING_TABLE,
) -> list[str]:
    """Apply pending files in filename order, each in its own transaction.

    Return applied filenames and open one connection. A failure rolls back
    only its migration. Missing or changed applied files raise
    `MigrationChecksumError`. Legacy rows with missing hashes are backfilled in
    their own transaction. The bookkeeping schema and table can be set for
    a sibling package.
    """
    _validate_bookkeeping_identifiers(
        bookkeeping_schema=bookkeeping_schema, bookkeeping_table=bookkeeping_table
    )
    conn = connect()
    try:
        _ensure_bookkeeping(
            conn, bookkeeping_schema=bookkeeping_schema, bookkeeping_table=bookkeeping_table
        )
        already = _applied_checksums(conn, bookkeeping_table=bookkeeping_table)
        discovered = discover_migrations(migrations_dir)
        on_disk = {path.name for path in discovered}
        missing = sorted(set(already) - on_disk)
        if missing:
            raise MigrationChecksumError(
                f"recorded as applied but no longer in {migrations_dir}: "
                f"{', '.join(missing)}; applied migrations stay in the tree "
                "-- fix forward with a new migration instead of removing or "
                "renaming one already applied"
            )
        applied: list[str] = []
        for path in discovered:
            version = path.name
            content_hash = _content_hash(path)
            if version in already:
                stored_hash = already[version]
                if stored_hash is None:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(
                            f"UPDATE {bookkeeping_table} SET content_sha256 = %s WHERE version = %s",
                            (content_hash, version),
                        )
                    finally:
                        cursor.close()
                    conn.commit()
                elif stored_hash != content_hash:
                    raise MigrationChecksumError(
                        f"{version} has changed since it was applied "
                        f"(recorded {stored_hash}, on disk {content_hash}); "
                        "applied migrations are immutable -- fix forward with a new migration"
                    )
                continue
            rendered = render(path.read_text(), substitutions=substitutions)
            cursor = conn.cursor()
            try:
                cursor.execute(rendered)
                cursor.execute(
                    f"INSERT INTO {bookkeeping_table} (version, content_sha256) VALUES (%s, %s)",
                    (version, content_hash),
                )
            finally:
                cursor.close()
            conn.commit()
            applied.append(version)
        return applied
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run `python -m dataset_store.migrate up`.

    Configuration and migration errors exit 1.
    """
    parser = argparse.ArgumentParser(prog="python -m dataset_store.migrate")
    parser.add_argument("command", choices=["up"])
    parser.add_argument("--bookkeeping-schema", default=_BOOKKEEPING_SCHEMA)
    parser.add_argument("--bookkeeping-table", default=_BOOKKEEPING_TABLE)
    args = parser.parse_args(argv)
    assert args.command == "up"

    try:
        settings = connection.ConnectionSettings.from_env(
            user_var="DATASET_STORE_MIGRATOR_DB_USER",
            password_var="DATASET_STORE_MIGRATOR_DB_PASSWORD",
            enable_iam_auth=False,
        )
        runtime_user = os.environ["DATASET_STORE_RUNTIME_DB_USER"]
        worker_user = os.environ["DATASET_STORE_WORKER_DB_USER"]
    except (DatasetStoreConfigError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    connector = connection.build_connector() if settings.instance_connection_name else None
    try:
        applied = apply_pending(
            connect=settings.connect_callable(connector=connector),
            substitutions={"RUNTIME_DB_USER": runtime_user, "WORKER_DB_USER": worker_user},
            bookkeeping_schema=args.bookkeeping_schema,
            bookkeeping_table=args.bookkeeping_table,
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
