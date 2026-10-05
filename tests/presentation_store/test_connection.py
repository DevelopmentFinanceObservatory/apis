"""Checks presentation store settings parsed from `PRESENTATION_STORE_*` environment variables."""

from __future__ import annotations

import pytest

from presentation_store.connection import PresentationStoreSettings
from presentation_store.errors import PresentationStoreConfigError


class TestPresentationStoreSettingsFromEnv:
    def test_returns_expected_settings_with_defaults(self, monkeypatch):
        monkeypatch.delenv("PRESENTATION_STORE_INSTANCE_CONNECTION_NAME", raising=False)
        monkeypatch.delenv("PRESENTATION_STORE_HOST", raising=False)
        monkeypatch.delenv("PRESENTATION_STORE_PORT", raising=False)
        monkeypatch.delenv("PRESENTATION_STORE_DATABASE", raising=False)
        monkeypatch.delenv("PRESENTATION_STORE_POOL_SIZE", raising=False)
        monkeypatch.delenv("PRESENTATION_STORE_CONNECT_TIMEOUT", raising=False)
        monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime@proj.iam")

        settings = PresentationStoreSettings.from_env(
            user_var="PRESENTATION_STORE_RUNTIME_DB_USER", enable_iam_auth=True
        )

        assert settings.instance_connection_name is None
        assert settings.host == "127.0.0.1"
        assert settings.port == 5432
        assert settings.database == "presentation"
        assert settings.user == "runtime@proj.iam"
        assert settings.password is None
        assert settings.enable_iam_auth is True
        assert settings.pool_size == 2
        assert settings.connect_timeout == 30.0

    def test_reads_migrator_user_and_password_pair(self, monkeypatch):
        monkeypatch.setenv("PRESENTATION_STORE_MIGRATOR_DB_USER", "migrator")
        monkeypatch.setenv("PRESENTATION_STORE_MIGRATOR_DB_PASSWORD", "secret")
        monkeypatch.setenv("PRESENTATION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")

        settings = PresentationStoreSettings.from_env(
            user_var="PRESENTATION_STORE_MIGRATOR_DB_USER",
            password_var="PRESENTATION_STORE_MIGRATOR_DB_PASSWORD",
        )

        assert settings.user == "migrator"
        assert settings.password == "secret"
        assert settings.instance_connection_name == "proj:region:inst"
        assert settings.enable_iam_auth is False

    def test_missing_user_var_raises_naming_it(self, monkeypatch):
        monkeypatch.delenv("PRESENTATION_STORE_RUNTIME_DB_USER", raising=False)

        with pytest.raises(
            PresentationStoreConfigError, match="PRESENTATION_STORE_RUNTIME_DB_USER"
        ):
            PresentationStoreSettings.from_env(user_var="PRESENTATION_STORE_RUNTIME_DB_USER")

    def test_missing_password_var_raises_naming_it(self, monkeypatch):
        monkeypatch.setenv("PRESENTATION_STORE_MIGRATOR_DB_USER", "migrator")
        monkeypatch.delenv("PRESENTATION_STORE_MIGRATOR_DB_PASSWORD", raising=False)

        with pytest.raises(
            PresentationStoreConfigError, match="PRESENTATION_STORE_MIGRATOR_DB_PASSWORD"
        ):
            PresentationStoreSettings.from_env(
                user_var="PRESENTATION_STORE_MIGRATOR_DB_USER",
                password_var="PRESENTATION_STORE_MIGRATOR_DB_PASSWORD",
            )

    def test_optional_password_var_ignored_once_the_connector_is_configured(self, monkeypatch):
        monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime@proj.iam")
        monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_PASSWORD", "localsecret")
        monkeypatch.setenv("PRESENTATION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")

        settings = PresentationStoreSettings.from_env(
            user_var="PRESENTATION_STORE_RUNTIME_DB_USER",
            optional_password_var="PRESENTATION_STORE_RUNTIME_DB_PASSWORD",
            enable_iam_auth=True,
        )
        assert settings.password is None
        assert settings.instance_connection_name == "proj:region:inst"

    def test_port_rejects_non_numeric_value(self, monkeypatch):
        monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("PRESENTATION_STORE_PORT", "not-a-port")

        with pytest.raises(PresentationStoreConfigError, match="PRESENTATION_STORE_PORT"):
            PresentationStoreSettings.from_env(user_var="PRESENTATION_STORE_RUNTIME_DB_USER")

    def test_pool_size_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("PRESENTATION_STORE_POOL_SIZE", "0")

        with pytest.raises(PresentationStoreConfigError, match="PRESENTATION_STORE_POOL_SIZE"):
            PresentationStoreSettings.from_env(user_var="PRESENTATION_STORE_RUNTIME_DB_USER")

    def test_connect_timeout_rejects_a_negative_value(self, monkeypatch):
        monkeypatch.setenv("PRESENTATION_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("PRESENTATION_STORE_CONNECT_TIMEOUT", "-5")

        with pytest.raises(
            PresentationStoreConfigError, match="PRESENTATION_STORE_CONNECT_TIMEOUT"
        ):
            PresentationStoreSettings.from_env(user_var="PRESENTATION_STORE_RUNTIME_DB_USER")


class TestConnectCallable:
    def test_calls_dataset_store_connect_with_settings_fields(self, monkeypatch):
        captured = {}

        def fake_connect(**kwargs):
            captured.update(kwargs)
            return object()

        # `connect_callable` is inherited from `dataset_store.connection
        # .BaseConnectionSettings`, so its own `connect` name resolves in
        # that module's globals regardless of which settings subclass
        # calls it -- patch it there, not in this module.
        monkeypatch.setattr("dataset_store.connection.connect", fake_connect)

        settings = PresentationStoreSettings(
            instance_connection_name="proj:region:inst",
            host="127.0.0.1",
            port=5432,
            database="presentation",
            user="runtime@proj.iam",
            password=None,
            enable_iam_auth=True,
            pool_size=2,
            connect_timeout=12.5,
        )
        settings.connect_callable(connector=None)()

        assert captured["database"] == "presentation"
        assert captured["user"] == "runtime@proj.iam"
        assert captured["enable_iam_auth"] is True
        assert captured["timeout"] == 12.5
