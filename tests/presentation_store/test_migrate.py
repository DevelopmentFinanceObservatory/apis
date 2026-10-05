"""Checks that the migration CLI uses presentation settings and migration metadata."""

from __future__ import annotations

import presentation_store.migrate as migrate_module
from presentation_store.migrate import MIGRATIONS_DIR, main

_ENV_VARS = (
    "PRESENTATION_STORE_MIGRATOR_DB_USER",
    "PRESENTATION_STORE_MIGRATOR_DB_PASSWORD",
    "PRESENTATION_STORE_RUNTIME_DB_USER",
)


def test_main_fails_fast_on_missing_env(monkeypatch, capsys):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    code = main(["up"])

    assert code == 1
    assert "ERROR" in capsys.readouterr().err


def test_main_is_wired_to_presentation_stores_own_dir_and_bookkeeping_table(monkeypatch, capsys):
    """The CLI passes the presentation migration directory and bookkeeping table."""
    captured: dict[str, object] = {}

    def _fake_apply_pending(
        *, connect, migrations_dir, substitutions, bookkeeping_schema, bookkeeping_table
    ):
        captured["migrations_dir"] = migrations_dir
        captured["bookkeeping_schema"] = bookkeeping_schema
        captured["bookkeeping_table"] = bookkeeping_table
        captured["substitutions"] = substitutions
        return []

    monkeypatch.setattr(migrate_module, "apply_pending", _fake_apply_pending)
    monkeypatch.setenv("PRESENTATION_STORE_MIGRATOR_DB_USER", "migrator")
    monkeypatch.setenv("PRESENTATION_STORE_MIGRATOR_DB_PASSWORD", "x")
    monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime")
    monkeypatch.delenv("PRESENTATION_STORE_INSTANCE_CONNECTION_NAME", raising=False)

    code = main(["up"])

    assert code == 0
    assert captured["migrations_dir"] == MIGRATIONS_DIR
    assert captured["bookkeeping_schema"] == "presentation"
    assert captured["bookkeeping_table"] == "presentation.presentation_store_schema_migrations"
    assert captured["substitutions"] == {"RUNTIME_DB_USER": "runtime"}
    assert "nothing to apply" in capsys.readouterr().out


def test_main_reports_applied_migrations(monkeypatch, capsys):
    monkeypatch.setattr(
        migrate_module,
        "apply_pending",
        lambda **kwargs: ["0001_bootstrap.sql", "0002_composition.sql"],
    )
    monkeypatch.setenv("PRESENTATION_STORE_MIGRATOR_DB_USER", "migrator")
    monkeypatch.setenv("PRESENTATION_STORE_MIGRATOR_DB_PASSWORD", "x")
    monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime")
    monkeypatch.delenv("PRESENTATION_STORE_INSTANCE_CONNECTION_NAME", raising=False)

    code = main(["up"])

    assert code == 0
    out = capsys.readouterr().out
    assert "0001_bootstrap.sql" in out
    assert "0002_composition.sql" in out
