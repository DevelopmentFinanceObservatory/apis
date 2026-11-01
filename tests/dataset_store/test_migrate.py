"""Test migrations offline with `RecordingConnection` and temporary SQL files.
Each migration run reuses one connection.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from dataset_store.migrate import (
    MIGRATIONS_DIR,
    MigrationChecksumError,
    MigrationTemplateError,
    apply_pending,
    discover_migrations,
    main,
    render,
)
from tests.dataset_store.conftest import RecordingConnection

_BOOKKEEPING_TABLE = "datasets.dataset_store_schema_migrations"


def _write(directory: Path, name: str, sql: str) -> Path:
    path = directory / name
    path.write_text(sql)
    return path


def test_apply_pending_applies_two_pending_migrations_in_filename_order(tmp_path: Path):
    _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    _write(tmp_path, "0002_y.sql", "CREATE TABLE y (id int);")
    conn = RecordingConnection()

    applied = apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    assert applied == ["0001_x.sql", "0002_y.sql"]
    statements = [sql for sql, _params in conn.executed]
    bookkeeping_index = statements.index("CREATE SCHEMA IF NOT EXISTS datasets")
    x_index = statements.index("CREATE TABLE x (id int);")
    y_index = statements.index("CREATE TABLE y (id int);")
    assert bookkeeping_index < x_index < y_index
    insert_indices = [i for i, sql in enumerate(statements) if sql.startswith("INSERT INTO")]
    assert len(insert_indices) == 2
    assert x_index < insert_indices[0] < y_index < insert_indices[1]


def test_apply_pending_is_idempotent_across_two_calls(tmp_path: Path):
    _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    _write(tmp_path, "0002_y.sql", "CREATE TABLE y (id int);")

    applied_rows: list[tuple[str, str]] = []

    def on_execute(sql: str, params) -> None:
        if sql.startswith(f"INSERT INTO {_BOOKKEEPING_TABLE}"):
            version, content_hash = params
            applied_rows.append((version, content_hash))
            conn.fetchall_result = list(applied_rows)

    conn = RecordingConnection(on_execute=on_execute)

    first = apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})
    second = apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    assert first == ["0001_x.sql", "0002_y.sql"]
    assert second == []
    statements = [sql for sql, _params in conn.executed]
    assert statements.count("CREATE TABLE x (id int);") == 1
    assert statements.count("CREATE TABLE y (id int);") == 1


def test_apply_pending_substitutes_runtime_db_user(tmp_path: Path):
    _write(
        tmp_path,
        "0001_grant.sql",
        'GRANT USAGE ON SCHEMA datasets TO "{{RUNTIME_DB_USER}}";',
    )
    conn = RecordingConnection()

    apply_pending(
        connect=lambda: conn,
        migrations_dir=tmp_path,
        substitutions={"RUNTIME_DB_USER": "alice"},
    )

    statements = [sql for sql, _params in conn.executed]
    assert 'GRANT USAGE ON SCHEMA datasets TO "alice";' in statements
    assert not any("{{RUNTIME_DB_USER}}" in sql for sql in statements)


def test_apply_pending_substitutes_worker_db_user(tmp_path: Path):
    _write(
        tmp_path,
        "0011_grant.sql",
        'GRANT "{{RUNTIME_DB_USER}}" TO "{{WORKER_DB_USER}}";',
    )
    conn = RecordingConnection()

    apply_pending(
        connect=lambda: conn,
        migrations_dir=tmp_path,
        substitutions={"RUNTIME_DB_USER": "alice", "WORKER_DB_USER": "bob"},
    )

    statements = [sql for sql, _params in conn.executed]
    assert 'GRANT "alice" TO "bob";' in statements
    assert not any("{{WORKER_DB_USER}}" in sql for sql in statements)


def test_render_substitutes_a_resolved_placeholder():
    assert render("hello {{NAME}}", substitutions={"NAME": "alice"}) == "hello alice"


def test_render_raises_on_unresolved_placeholder():
    with pytest.raises(MigrationTemplateError, match="NAME"):
        render("hello {{NAME}}", substitutions={})


def test_render_raises_on_substitution_value_with_invalid_characters():
    with pytest.raises(MigrationTemplateError):
        render("hello {{NAME}}", substitutions={"NAME": 'alice"; DROP--'})


def test_render_accepts_a_realistic_service_account_email():
    value = "dfo-dev-apis-runtime@dev-fin-observatory.iam"
    assert render("{{NAME}}", substitutions={"NAME": value}) == value


def test_mid_file_failure_rolls_back_and_is_retried_on_next_call(tmp_path: Path):
    _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    should_fail = True

    def on_execute(sql: str, params) -> None:
        if should_fail and sql == "CREATE TABLE x (id int);":
            raise RuntimeError("boom")

    conn = RecordingConnection(on_execute=on_execute)

    with pytest.raises(RuntimeError):
        apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})
    assert conn.rollbacks == 1

    should_fail = False
    applied = apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})
    assert applied == ["0001_x.sql"]


def test_apply_pending_lets_a_mid_file_exception_propagate_unwrapped(tmp_path: Path):
    """Propagate driver exceptions unwrapped. `apply_pending` owns rollback."""
    _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")

    def on_execute(sql: str, params) -> None:
        if sql == "CREATE TABLE x (id int);":
            raise RuntimeError("boom")

    conn = RecordingConnection(on_execute=on_execute)

    with pytest.raises(RuntimeError, match="boom"):
        apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})


def test_discover_migrations_sorts_by_filename_regardless_of_listing_order(tmp_path: Path):
    _write(tmp_path, "0002_y.sql", "-- y")
    _write(tmp_path, "0001_x.sql", "-- x")

    result = discover_migrations(tmp_path)

    assert [p.name for p in result] == ["0001_x.sql", "0002_y.sql"]


def test_apply_pending_uses_alternate_bookkeeping_identifiers_when_supplied(tmp_path: Path):
    """Allow another store package to keep migration records in its own schema and table."""
    _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    conn = RecordingConnection()
    alt_table = "presentation.presentation_store_schema_migrations"

    applied = apply_pending(
        connect=lambda: conn,
        migrations_dir=tmp_path,
        substitutions={},
        bookkeeping_schema="presentation",
        bookkeeping_table=alt_table,
    )

    assert applied == ["0001_x.sql"]
    statements = [sql for sql, _params in conn.executed]
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS presentation"
    assert statements[1].startswith(f"CREATE TABLE IF NOT EXISTS {alt_table}")
    assert any(sql.startswith(f"INSERT INTO {alt_table}") for sql in statements)
    assert not any("datasets.dataset_store_schema_migrations" in sql for sql in statements)


def test_apply_pending_rejects_an_invalid_bookkeeping_schema(tmp_path: Path):
    conn = RecordingConnection()

    with pytest.raises(ValueError, match="bookkeeping_schema"):
        apply_pending(
            connect=lambda: conn,
            migrations_dir=tmp_path,
            substitutions={},
            bookkeeping_schema="bad schema; DROP TABLE x",
        )
    assert conn.executed == []


def test_apply_pending_rejects_an_invalid_bookkeeping_table(tmp_path: Path):
    conn = RecordingConnection()

    with pytest.raises(ValueError, match="bookkeeping_table"):
        apply_pending(
            connect=lambda: conn,
            migrations_dir=tmp_path,
            substitutions={},
            bookkeeping_table="datasets.migrations; DROP TABLE x",
        )
    assert conn.executed == []


def test_main_accepts_bookkeeping_flags_and_still_fails_fast_on_missing_env(monkeypatch, capsys):
    for var in (
        "DATASET_STORE_MIGRATOR_DB_USER",
        "DATASET_STORE_MIGRATOR_DB_PASSWORD",
        "DATASET_STORE_RUNTIME_DB_USER",
    ):
        monkeypatch.delenv(var, raising=False)

    # The missing-environment error proves both bookkeeping flags parsed.
    code = main(
        ["up", "--bookkeeping-schema", "custom", "--bookkeeping-table", "custom.migrations"]
    )

    assert code == 1
    assert "ERROR" in capsys.readouterr().err


def test_ensure_bookkeeping_creates_schema_then_table_before_any_migration(tmp_path: Path):
    _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    conn = RecordingConnection()

    apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    first_two = [sql for sql, _params in conn.executed[:2]]
    assert first_two[0] == "CREATE SCHEMA IF NOT EXISTS datasets"
    assert first_two[1].startswith(f"CREATE TABLE IF NOT EXISTS {_BOOKKEEPING_TABLE}")


def test_apply_pending_records_the_on_disk_content_hash(tmp_path: Path):
    path = _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    conn = RecordingConnection()

    apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    insert_calls = [
        params
        for sql, params in conn.executed
        if sql.startswith(f"INSERT INTO {_BOOKKEEPING_TABLE}")
    ]
    assert len(insert_calls) == 1
    version, content_hash = insert_calls[0]
    assert version == "0001_x.sql"
    assert content_hash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_modifying_an_already_applied_file_raises_checksum_error(tmp_path: Path):
    path = _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    conn = RecordingConnection(fetchall_result=[("0001_x.sql", original_hash)])

    path.write_text("CREATE TABLE x (id int, extra int);")

    with pytest.raises(MigrationChecksumError, match=re.escape("0001_x.sql")):
        apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    assert conn.rollbacks == 1


def test_unmodified_already_applied_file_does_not_raise_or_reapply(tmp_path: Path):
    path = _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    conn = RecordingConnection(fetchall_result=[("0001_x.sql", content_hash)])

    applied = apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    assert applied == []
    assert not any(sql == "CREATE TABLE x (id int);" for sql, _params in conn.executed)


def test_legacy_row_without_a_hash_is_backfilled_then_a_later_change_raises(tmp_path: Path):
    path = _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    conn = RecordingConnection(fetchall_result=[("0001_x.sql", None)])

    applied = apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    assert applied == []
    update_calls = [
        (sql, params)
        for sql, params in conn.executed
        if sql.startswith(f"UPDATE {_BOOKKEEPING_TABLE}")
    ]
    assert len(update_calls) == 1
    _sql, (backfilled_hash, backfilled_version) = update_calls[0]
    assert backfilled_version == "0001_x.sql"
    assert backfilled_hash == original_hash

    # A later file edit must fail against the hash backfilled above.
    conn.fetchall_result = [("0001_x.sql", original_hash)]
    path.write_text("CREATE TABLE x (id int, extra int);")

    with pytest.raises(MigrationChecksumError, match=re.escape("0001_x.sql")):
        apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})


def test_deleting_an_already_applied_file_raises_naming_it(tmp_path: Path):
    path = _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    conn = RecordingConnection(fetchall_result=[("0001_x.sql", original_hash)])

    path.unlink()

    with pytest.raises(MigrationChecksumError, match=re.escape("0001_x.sql")):
        apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})


def test_renaming_an_already_applied_file_raises_for_the_old_name(tmp_path: Path):
    path = _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    conn = RecordingConnection(fetchall_result=[("0001_x.sql", original_hash)])

    path.rename(tmp_path / "0001_x_renamed.sql")

    with pytest.raises(MigrationChecksumError, match=re.escape("0001_x.sql")):
        apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})


def test_apply_pending_applies_every_real_migration_in_order():
    """Apply real migrations offline to check placeholders, order, and checksums.

    Postgres DDL validity is covered by the live tests.
    """
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
        substitutions={
            "RUNTIME_DB_USER": "test-runtime-user",
            "WORKER_DB_USER": "test-worker-user",
        },
    )

    assert applied == [
        "0001_bootstrap.sql",
        "0002_spec_registry.sql",
        "0003_artifact_inventory.sql",
        "0004_version_records.sql",
        "0006_version_revocation.sql",
        "0007_artifact_references.sql",
        "0008_artifact_location_tier_index.sql",
        "0009_subject_head.sql",
        "0010_version_record_base_content_version.sql",
        "0011_operation.sql",
        "0012_worker_role.sql",
        "0013_operation_retryable_and_index.sql",
        "0014_operation_dispatch.sql",
        "0015_operation_origin_catalogue_build.sql",
        "0016_operation_refusal.sql",
        "0017_trace_conversion_entries.sql",
        "0018_backfill_prepared_at.sql",
        "0019_backfill_prepared_at_after_deploy.sql",
        "0020_trace_conversion_entries_after_deploy.sql",
        "0021_drop_collections.sql",
        "0022_drop_null_collection.sql",
        "0023_frame_recipe.sql",
        "0024_operation_attempt_fencing.sql",
        "0025_artifact_location_renewal.sql",
        "0026_artifact_reference_principal_label.sql",
        "0027_materialisation_record.sql",
        "0028_spec_alias.sql",
        "0029_version_record_frame_grammar_version.sql",
        "0030_version_record_facts.sql",
        "0031_version_record_facts_backfill.sql",
        "0032_operation_confirmed_at.sql",
        "0033_output_profile_party_kind.sql",
        "0034_description_record.sql",
        "0035_strip_backfilled_documentation.sql",
        "0036_version_record_continuations.sql",
        "0037_version_record_recency.sql",
        "0038_version_record_frame_summed.sql",
        "0039_caller_allowance.sql",
        "0040_caller_allowance_budgets.sql",
        "0041_analysis_answer.sql",
        "0042_caller_allowance_card.sql",
        "0043_frame_recipe_superseded.sql",
        "0044_export.sql",
        "0045_copy_claim.sql",
        "0046_head_family.sql",
        "0047_warning_counted_refs.sql",
    ]
    statements = [sql for sql, _params in conn.executed]
    assert any("CREATE TABLE datasets.version_record" in sql for sql in statements)
    assert any("CREATE TABLE datasets.version_member" in sql for sql in statements)
    assert any("DROP TABLE IF EXISTS datasets.version_member" in sql for sql in statements)
    assert any("CREATE TABLE datasets.version_artifact" in sql for sql in statements)
    assert any("CREATE TABLE datasets.frame_recipe" in sql for sql in statements)
    assert any("CREATE TABLE datasets.materialisation" in sql for sql in statements)
    assert any("CREATE TABLE datasets.spec_alias" in sql for sql in statements)
    assert any("CREATE TABLE datasets.description_record" in sql for sql in statements)
    assert any("CREATE TABLE datasets.version_revocation" in sql for sql in statements)
    assert any("CREATE TABLE datasets.artifact_reference" in sql for sql in statements)
    assert any("CREATE TABLE datasets.caller_allowance" in sql for sql in statements)
    assert any("CREATE TABLE IF NOT EXISTS datasets.analysis_answer" in sql for sql in statements)
    assert any("CREATE TABLE IF NOT EXISTS datasets.export" in sql for sql in statements)
    assert any(
        "CREATE INDEX artifact_location_tier_idx ON datasets.artifact_location" in sql
        for sql in statements
    )
    assert any("CREATE TABLE datasets.subject_head" in sql for sql in statements)
    assert any(
        "ALTER TABLE datasets.artifact_location DROP CONSTRAINT artifact_location_check" in sql
        for sql in statements
    )
    assert any(
        "ALTER TABLE datasets.artifact_location ADD COLUMN renewal" in sql for sql in statements
    )
    assert any(
        "ALTER TABLE datasets.version_record ADD COLUMN base_content_version" in sql
        for sql in statements
    )
    assert any("CREATE TABLE datasets.operation" in sql for sql in statements)
    # The fencing migration is additive, so the image before it runs against the migrated table.
    fencing = [sql for sql in statements if "ADD COLUMN claimed_at timestamptz" in sql]
    assert len(fencing) == 1
    assert "DROP " not in fencing[0]
    assert "CONSTRAINT" not in fencing[0]
    assert any(
        "CREATE INDEX operation_backlog_state_idx ON datasets.operation" in sql
        for sql in statements
    )
    assert any(
        "UPDATE datasets.version_record" in sql
        and "'{trace,conversion}'" in sql
        and "jsonb_typeof(immutable_body -> 'trace' -> 'conversion') = 'object'" in sql
        for sql in statements
    )
    # A second pass verifies checksums without reapplying migrations.
    second = apply_pending(
        connect=lambda: conn,
        migrations_dir=MIGRATIONS_DIR,
        substitutions={"RUNTIME_DB_USER": "test-runtime-user"},
    )
    assert second == []


def test_missing_recorded_file_is_caught_before_a_pending_file_would_apply(tmp_path: Path):
    path = _write(tmp_path, "0001_x.sql", "CREATE TABLE x (id int);")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(tmp_path, "0002_y.sql", "CREATE TABLE y (id int);")
    conn = RecordingConnection(fetchall_result=[("0001_x.sql", original_hash)])

    path.unlink()

    with pytest.raises(MigrationChecksumError, match=re.escape("0001_x.sql")):
        apply_pending(connect=lambda: conn, migrations_dir=tmp_path, substitutions={})

    assert not any(sql == "CREATE TABLE y (id int);" for sql, _params in conn.executed)
    assert conn.rollbacks == 1


def _statement_kinds(name: str) -> list[str]:
    """The first two words of each statement of `name`, comments dropped and each
    dollar-quoted body read as `$$`."""
    code = "\n".join(
        line
        for line in (MIGRATIONS_DIR / name).read_text().splitlines()
        if not line.lstrip().startswith("--")
    )
    code = re.sub(r"\$(\w*)\$.*?\$\1\$", "$$", code, flags=re.DOTALL)
    return [" ".join(statement.split()[:2]) for statement in code.split(";") if statement.strip()]


@pytest.mark.parametrize(
    ("name", "kinds"),
    [
        ("0029_version_record_frame_grammar_version.sql", ["SET LOCAL", "ALTER TABLE"]),
        (
            "0030_version_record_facts.sql",
            ["SET LOCAL", "ALTER TABLE", "CREATE OR", "CREATE OR"],
        ),
        ("0031_version_record_facts_backfill.sql", ["UPDATE datasets.version_record", "DO $$"]),
        ("0034_description_record.sql", ["SET LOCAL", "CREATE TABLE"]),
        ("0037_version_record_recency.sql", ["SET LOCAL", "CREATE INDEX"]),
        ("0038_version_record_frame_summed.sql", ["SET LOCAL", "ALTER TABLE"]),
        (
            "0033_output_profile_party_kind.sql",
            [
                "SET LOCAL",
                "CREATE OR",
                "CREATE OR",
                "UPDATE datasets.version_record",
                "UPDATE datasets.version_record",
            ],
        ),
    ],
)
def test_version_record_alters_run_apart_from_the_backfill(name: str, kinds: list[str]):
    """An ALTER holds ACCESS EXCLUSIVE until its transaction commits, so a file that alters
    `version_record` while the previous image serves writes no row and gives up waiting for
    the lock, and the backfill and its closing check run in their own file under row locks."""
    assert _statement_kinds(name) == kinds
    if kinds[0] == "SET LOCAL":
        assert "SET LOCAL lock_timeout = '5s'" in (MIGRATIONS_DIR / name).read_text()
