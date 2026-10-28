"""Checks migration order, bookkeeping, SQL substitutions, and idempotency offline."""

from __future__ import annotations

import json
import re
from pathlib import Path

from assembler.catalogue import load_release
from dataset_store.migrate import apply_pending
from tests.dataset_store.conftest import RecordingConnection

MIGRATIONS_DIR = Path(__file__).parents[2] / "src" / "presentation_store" / "migrations"
_BOOKKEEPING_TABLE = "presentation.presentation_store_schema_migrations"


def test_presentation_store_migrations_apply_in_order_and_are_idempotent():
    applied_rows: list[tuple[str, str]] = []

    def on_execute(sql: str, params) -> None:
        if sql.startswith(f"INSERT INTO {_BOOKKEEPING_TABLE}"):
            version, content_hash = params
            applied_rows.append((version, content_hash))
            conn.fetchall_result = list(applied_rows)

    conn = RecordingConnection(on_execute=on_execute)

    applied = apply_pending(
        connect=lambda: conn,
        migrations_dir=MIGRATIONS_DIR,
        substitutions={"RUNTIME_DB_USER": "test-runtime-user"},
        bookkeeping_schema="presentation",
        bookkeeping_table=_BOOKKEEPING_TABLE,
    )

    assert applied == [
        "0001_bootstrap.sql",
        "0002_composition.sql",
        "0003_component.sql",
        "0004_draft_lease.sql",
        "0005_series_coordinates.sql",
        "0006_bar_chart_axes.sql",
        "0007_write_token.sql",
        "0008_idempotent_create.sql",
        "0009_drop_draft_lease.sql",
        "0010_revision_formatter_profile.sql",
        "0011_revision_frame_rendering.sql",
        "0012_share.sql",
        "0013_share_slot_check.sql",
    ]
    statements = [sql for sql, _params in conn.executed]
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS presentation"
    assert any("CREATE TABLE presentation.composition (" in sql for sql in statements)
    assert any("CREATE TABLE presentation.composition_revision (" in sql for sql in statements)
    assert any("CREATE TABLE presentation.component (" in sql for sql in statements)
    assert any("CREATE TABLE presentation.component_revision (" in sql for sql in statements)
    for table in ("share", "share_revision", "share_slot", "served_frame", "share_slot_check"):
        assert any(f"CREATE TABLE presentation.{table} (" in sql for sql in statements), table
    assert any("CREATE TABLE presentation.draft_lease (" in sql for sql in statements)
    assert any(
        "CREATE INDEX draft_lease_expires_at_idx ON presentation.draft_lease" in sql
        for sql in statements
    )
    assert any(
        "ALTER TABLE presentation.component_revision ADD COLUMN series_coordinates jsonb" in sql
        for sql in statements
    )
    assert any(
        "UPDATE presentation.component_revision" in sql
        and "category_axis" in sql
        and "value_axis" in sql
        for sql in statements
    )
    assert any(
        "ALTER TABLE presentation.component ADD COLUMN write_token_sha256 bytea" in sql
        for sql in statements
    )
    assert any(
        "ALTER TABLE presentation.composition ADD COLUMN write_token_sha256 bytea" in sql
        for sql in statements
    )
    assert any(
        "CREATE UNIQUE INDEX component_write_token_sha256_key" in sql
        and "ON presentation.component (write_token_sha256)" in sql
        for sql in statements
    )
    assert any(
        "ALTER TABLE presentation.component ADD COLUMN create_request_sha256 bytea" in sql
        for sql in statements
    )
    assert any("DROP TABLE IF EXISTS presentation.draft_lease" in sql for sql in statements)
    assert any(
        "ADD COLUMN formatter_profile jsonb" in sql
        and "ALTER COLUMN formatter_profile SET NOT NULL" in sql
        for sql in statements
    )
    assert not any("{{" in sql for sql in statements)
    assert not any("datasets.dataset_store_schema_migrations" in sql for sql in statements)

    second = apply_pending(
        connect=lambda: conn,
        migrations_dir=MIGRATIONS_DIR,
        substitutions={"RUNTIME_DB_USER": "test-runtime-user"},
        bookkeeping_schema="presentation",
        bookkeeping_table=_BOOKKEEPING_TABLE,
    )
    assert second == []


def test_bar_chart_axes_migration_applies_after_the_column_it_rewrites():
    """Migration 0006 rewrites the `resolved_payload` column created in 0003."""
    names = sorted(path.name for path in MIGRATIONS_DIR.glob("*.sql"))
    assert names.index("0003_component.sql") < names.index("0006_bar_chart_axes.sql")


def test_bar_chart_axes_migration_is_a_data_rewrite_scoped_to_bar_chart_rows():
    """The SQL renames axes only on legacy bar charts and guards on `x_axis`."""
    sql_text = (MIGRATIONS_DIR / "0006_bar_chart_axes.sql").read_text()
    assert "UPDATE presentation.component_revision" in sql_text
    assert "resolved_payload -> 'component' ->> 'type' = 'bar_chart'" in sql_text
    assert "category_axis" in sql_text
    assert "value_axis" in sql_text
    assert "resolved_payload -> 'component' ? 'x_axis'" in sql_text


def test_formatter_profile_migration_writes_the_2026_8_0_profile():
    """0010's backfill and default both carry the profile the installed release still ships.

    The migration's literal is the 2026.8.0 profile. Release 2026.10.0 left the profile
    unchanged, so the installed release is a valid witness. A release that changes the
    profile fails this test, and the check then needs a frozen copy of the 2026.8.0 profile.
    """
    release = load_release()
    sql_text = (MIGRATIONS_DIR / "0010_revision_formatter_profile.sql").read_text()
    literals = re.findall(r"'(\{.*\})'::jsonb", sql_text)
    assert len(literals) == 2
    assert [json.loads(literal) for literal in literals] == [dict(release.formatter_profile)] * 2
    assert "WHERE catalog_version = '2026.8.0'" in sql_text
