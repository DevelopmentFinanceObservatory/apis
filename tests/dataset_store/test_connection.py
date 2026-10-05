"""Test `dataset_store.connection`; pg8000 has no injection hook for plain TCP."""

from __future__ import annotations

import threading
import time

import pg8000
import pg8000.exceptions
import pytest

from dataset_store.connection import ConnectionSettings, connect, default_is_broken
from dataset_store.errors import DatasetStoreConfigError, DatasetStoreConnectionError


class _FakeUsock:
    """Stands in for pg8000's private `_usock` socket attribute."""

    def __init__(self) -> None:
        self.settimeout_calls: list[float] = []

    def settimeout(self, value: float) -> None:
        self.settimeout_calls.append(value)


class _FakeConnection:
    def __init__(self) -> None:
        self._usock = _FakeUsock()


class _FakeConnector:
    def __init__(
        self, *, raises: Exception | None = None, connection: object | None = None
    ) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self._raises = raises
        self.sentinel = connection if connection is not None else _FakeConnection()

    def connect(self, instance_connection_string: str, driver: str, **kwargs: object):
        self.calls.append((instance_connection_string, driver, kwargs))
        if self._raises is not None:
            raise self._raises
        return self.sentinel


def test_connector_path_iam_auth_sends_no_password_key():
    fake = _FakeConnector()
    connect(
        connector=fake,
        instance_connection_name="proj:region:inst",
        host="unused",
        port=5432,
        database="datasets",
        user="runtime@proj.iam",
        password=None,
        enable_iam_auth=True,
    )
    (instance, driver, kwargs) = fake.calls[0]
    assert instance == "proj:region:inst"
    assert driver == "pg8000"
    assert kwargs["enable_iam_auth"] is True
    assert "password" not in kwargs


def test_connector_path_migrator_auth_sends_password():
    fake = _FakeConnector()
    connect(
        connector=fake,
        instance_connection_name="proj:region:inst",
        host="unused",
        port=5432,
        database="datasets",
        user="migrator",
        password="secret",
        enable_iam_auth=False,
    )
    (_, _, kwargs) = fake.calls[0]
    assert kwargs["password"] == "secret"
    assert kwargs["enable_iam_auth"] is False


def test_plain_tcp_branch_calls_pg8000_connect_with_expected_args(monkeypatch):
    calls = []

    def fake_pg8000_connect(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(pg8000, "connect", fake_pg8000_connect)

    connect(
        connector=None,
        instance_connection_name=None,
        host="127.0.0.1",
        port=5432,
        database="datasets",
        user="postgres",
        password="x",
    )
    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 5432
    assert calls[0]["user"] == "postgres"
    assert calls[0]["password"] == "x"
    assert calls[0]["database"] == "datasets"


def test_plain_tcp_with_no_password_raises_before_calling_pg8000(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("pg8000.connect must not be called")

    monkeypatch.setattr(pg8000, "connect", fail_if_called)

    with pytest.raises(DatasetStoreConnectionError):
        connect(
            connector=None,
            instance_connection_name=None,
            host="127.0.0.1",
            port=5432,
            database="datasets",
            user="postgres",
            password=None,
        )


def test_connector_failure_is_wrapped_with_original_as_cause():
    original = RuntimeError("auth failed")
    fake = _FakeConnector(raises=original)

    with pytest.raises(DatasetStoreConnectionError) as excinfo:
        connect(
            connector=fake,
            instance_connection_name="proj:region:inst",
            host="unused",
            port=5432,
            database="datasets",
            user="runtime@proj.iam",
            enable_iam_auth=True,
        )
    assert excinfo.value.__cause__ is original


def test_connector_path_sets_timeout_on_the_usock():
    fake = _FakeConnector()
    connect(
        connector=fake,
        instance_connection_name="proj:region:inst",
        host="unused",
        port=5432,
        database="datasets",
        user="runtime@proj.iam",
        password=None,
        enable_iam_auth=True,
        timeout=12.5,
    )
    assert isinstance(fake.sentinel, _FakeConnection)
    assert fake.sentinel._usock.settimeout_calls == [12.5]


def test_connector_path_connection_without_usock_raises():
    class _NoUsockConnection:
        pass

    fake = _FakeConnector(connection=_NoUsockConnection())

    with pytest.raises(DatasetStoreConnectionError):
        connect(
            connector=fake,
            instance_connection_name="proj:region:inst",
            host="unused",
            port=5432,
            database="datasets",
            user="runtime@proj.iam",
            password=None,
            enable_iam_auth=True,
        )


def test_connector_path_hung_connect_raises_promptly_instead_of_blocking():
    release = threading.Event()

    class _HangingConnector:
        def connect(self, instance_connection_string: str, driver: str, **kwargs: object):
            # Bound the wait so a failed timeout guard cannot strand a worker after the test.
            release.wait(timeout=5)
            return _FakeConnection()

    try:
        with pytest.raises(DatasetStoreConnectionError):
            connect(
                connector=_HangingConnector(),
                instance_connection_name="proj:region:inst",
                host="unused",
                port=5432,
                database="datasets",
                user="runtime@proj.iam",
                password=None,
                enable_iam_auth=True,
                timeout=0.1,
            )
    finally:
        release.set()


def test_connector_path_late_completing_dial_closes_the_connection():
    release = threading.Event()

    class _FakeCloseableConnection:
        def __init__(self) -> None:
            self._usock = _FakeUsock()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    conn = _FakeCloseableConnection()

    class _LateConnector:
        def connect(self, instance_connection_string: str, driver: str, **kwargs: object):
            release.wait(timeout=5)
            return conn

    with pytest.raises(DatasetStoreConnectionError):
        connect(
            connector=_LateConnector(),
            instance_connection_name="proj:region:inst",
            host="unused",
            port=5432,
            database="datasets",
            user="runtime@proj.iam",
            password=None,
            enable_iam_auth=True,
            timeout=0.05,
        )

    release.set()
    deadline = time.monotonic() + 5
    while conn.close_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert conn.close_calls == 1


def test_connector_path_on_time_success_does_not_close_the_connection():
    class _FakeCloseableConnection:
        def __init__(self) -> None:
            self._usock = _FakeUsock()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    sentinel = _FakeCloseableConnection()
    fake = _FakeConnector(connection=sentinel)

    conn = connect(
        connector=fake,
        instance_connection_name="proj:region:inst",
        host="unused",
        port=5432,
        database="datasets",
        user="runtime@proj.iam",
        password=None,
        enable_iam_auth=True,
    )
    assert conn is sentinel
    assert sentinel.close_calls == 0


class TestDefaultIsBroken:
    def test_true_for_interface_error(self):
        assert default_is_broken(pg8000.exceptions.InterfaceError("dead")) is True

    def test_true_for_bare_os_error(self):
        assert default_is_broken(OSError("socket died")) is True

    def test_false_for_value_error(self):
        assert default_is_broken(ValueError("not a transport failure")) is False

    def test_true_for_dataset_store_connection_error(self):
        assert default_is_broken(DatasetStoreConnectionError("x")) is True


class TestConnectionSettingsFromEnv:
    def test_returns_expected_settings_with_defaults(self, monkeypatch):
        monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
        monkeypatch.delenv("DATASET_STORE_HOST", raising=False)
        monkeypatch.delenv("DATASET_STORE_PORT", raising=False)
        monkeypatch.delenv("DATASET_STORE_DATABASE", raising=False)
        monkeypatch.delenv("DATASET_STORE_POOL_SIZE", raising=False)
        monkeypatch.delenv("DATASET_STORE_CONNECT_TIMEOUT", raising=False)
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime@proj.iam")

        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_RUNTIME_DB_USER",
            enable_iam_auth=True,
        )

        assert settings.instance_connection_name is None
        assert settings.host == "127.0.0.1"
        assert settings.port == 5432
        assert settings.database == "datasets"
        assert settings.user == "runtime@proj.iam"
        assert settings.password is None
        assert settings.enable_iam_auth is True
        assert settings.pool_size == 10
        assert settings.connect_timeout == 30.0

    def test_reads_connect_timeout_from_its_env_var(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime@proj.iam")
        monkeypatch.setenv("DATASET_STORE_CONNECT_TIMEOUT", "12.5")

        settings = ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

        assert settings.connect_timeout == 12.5

    def test_reads_migrator_user_and_password_pair(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_MIGRATOR_DB_USER", "migrator")
        monkeypatch.setenv("DATASET_STORE_MIGRATOR_DB_PASSWORD", "secret")
        monkeypatch.setenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")
        monkeypatch.setenv("DATASET_STORE_POOL_SIZE", "5")

        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_MIGRATOR_DB_USER",
            password_var="DATASET_STORE_MIGRATOR_DB_PASSWORD",
        )

        assert settings.user == "migrator"
        assert settings.password == "secret"
        assert settings.instance_connection_name == "proj:region:inst"
        assert settings.pool_size == 5
        assert settings.enable_iam_auth is False

    def test_missing_user_var_raises_naming_it(self, monkeypatch):
        monkeypatch.delenv("DATASET_STORE_RUNTIME_DB_USER", raising=False)

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_RUNTIME_DB_USER"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

    def test_missing_password_var_raises_naming_it(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_MIGRATOR_DB_USER", "migrator")
        monkeypatch.delenv("DATASET_STORE_MIGRATOR_DB_PASSWORD", raising=False)

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_MIGRATOR_DB_PASSWORD"):
            ConnectionSettings.from_env(
                user_var="DATASET_STORE_MIGRATOR_DB_USER",
                password_var="DATASET_STORE_MIGRATOR_DB_PASSWORD",
            )

    def test_optional_password_var_reads_it_when_set(self, monkeypatch):
        monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_PASSWORD", "localsecret")

        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_RUNTIME_DB_USER",
            optional_password_var="DATASET_STORE_RUNTIME_DB_PASSWORD",
        )
        assert settings.password == "localsecret"

    @pytest.mark.parametrize(
        ("password_options", "password_env_value"),
        [
            pytest.param(
                {"password_var": None},
                "must-be-ignored",
                id="disabled-password-variable",
            ),
            pytest.param(
                {"optional_password_var": "DATASET_STORE_RUNTIME_DB_PASSWORD"},
                None,
                id="optional-password-variable-unset",
            ),
        ],
    )
    def test_password_settings_without_a_password_return_none(
        self, monkeypatch, password_options, password_env_value
    ):
        monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime@proj.iam")
        if password_env_value is None:
            monkeypatch.delenv("DATASET_STORE_RUNTIME_DB_PASSWORD", raising=False)
        else:
            monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_PASSWORD", password_env_value)

        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_RUNTIME_DB_USER",
            **password_options,
        )
        assert settings.password is None

    def test_optional_password_var_ignored_once_the_connector_is_configured(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime@proj.iam")
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_PASSWORD", "localsecret")
        monkeypatch.setenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")

        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_RUNTIME_DB_USER",
            optional_password_var="DATASET_STORE_RUNTIME_DB_PASSWORD",
            enable_iam_auth=True,
        )
        assert settings.password is None
        assert settings.instance_connection_name == "proj:region:inst"


class TestConnectionSettingsNumericValidation:
    def test_port_rejects_non_numeric_value(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_PORT", "not-a-port")

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_PORT"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

    def test_port_accepts_a_valid_value(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_PORT", "6543")

        settings = ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")
        assert settings.port == 6543

    def test_pool_size_rejects_non_numeric_value(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_POOL_SIZE", "lots")

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_POOL_SIZE"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

    def test_pool_size_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_POOL_SIZE", "0")

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_POOL_SIZE"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

    def test_pool_size_accepts_a_valid_value(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_POOL_SIZE", "4")

        settings = ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")
        assert settings.pool_size == 4

    def test_connect_timeout_rejects_non_numeric_value(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_CONNECT_TIMEOUT", "soon")

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_CONNECT_TIMEOUT"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

    def test_connect_timeout_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_CONNECT_TIMEOUT", "0")

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_CONNECT_TIMEOUT"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

    def test_connect_timeout_rejects_a_negative_value(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_CONNECT_TIMEOUT", "-5")

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_CONNECT_TIMEOUT"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")

    def test_connect_timeout_accepts_a_valid_value(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_CONNECT_TIMEOUT", "45")

        settings = ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")
        assert settings.connect_timeout == 45.0

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
    def test_connect_timeout_rejects_non_finite_values(self, monkeypatch, raw):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_CONNECT_TIMEOUT", raw)

        with pytest.raises(DatasetStoreConfigError, match="DATASET_STORE_CONNECT_TIMEOUT"):
            ConnectionSettings.from_env(user_var="DATASET_STORE_RUNTIME_DB_USER")


class TestConnectCallable:
    def test_calls_connect_with_settings_fields(self, monkeypatch):
        captured = {}

        def fake_connect(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("dataset_store.connection.connect", fake_connect)

        settings = ConnectionSettings(
            instance_connection_name="proj:region:inst",
            host="127.0.0.1",
            port=5432,
            database="datasets",
            user="runtime@proj.iam",
            password=None,
            enable_iam_auth=True,
            pool_size=2,
            connect_timeout=12.5,
        )
        connector = _FakeConnector()
        settings.connect_callable(connector=connector)()

        assert captured == {
            "connector": connector,
            "instance_connection_name": "proj:region:inst",
            "host": "127.0.0.1",
            "port": 5432,
            "database": "datasets",
            "user": "runtime@proj.iam",
            "password": None,
            "enable_iam_auth": True,
            "timeout": 12.5,
        }

    def test_plain_tcp_wiring_threads_the_optional_runtime_password_through(self, monkeypatch):
        monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime")
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_PASSWORD", "localsecret")

        captured = {}

        def fake_connect(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(pg8000, "connect", fake_connect)

        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_RUNTIME_DB_USER",
            optional_password_var="DATASET_STORE_RUNTIME_DB_PASSWORD",
        )
        settings.connect_callable(connector=None)()

        assert captured["password"] == "localsecret"

    def test_connector_wiring_still_does_iam_with_no_password_when_var_is_set(self, monkeypatch):
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_USER", "runtime@proj.iam")
        monkeypatch.setenv("DATASET_STORE_RUNTIME_DB_PASSWORD", "localsecret")
        monkeypatch.setenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", "proj:region:inst")

        settings = ConnectionSettings.from_env(
            user_var="DATASET_STORE_RUNTIME_DB_USER",
            optional_password_var="DATASET_STORE_RUNTIME_DB_PASSWORD",
            enable_iam_auth=True,
        )
        fake = _FakeConnector()
        settings.connect_callable(connector=fake)()

        (_, _, kwargs) = fake.calls[0]
        assert kwargs["enable_iam_auth"] is True
        assert "password" not in kwargs
