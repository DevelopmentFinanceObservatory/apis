"""Fake ADC credentials for `CloudRunGenerationSource` tests, and a fake served catalogue."""

from __future__ import annotations

from collections.abc import Callable


class FakeCredentials:
    """Fake credentials with configurable refresh results and call count."""

    def __init__(
        self,
        *,
        refreshed_token: str | None = "fake-bearer-token",
        valid: bool = False,
    ) -> None:
        self.token: str | None = None
        self.valid = valid
        self._refreshed_token = refreshed_token
        self.refresh_calls = 0

    def refresh(self, request: object) -> None:
        self.refresh_calls += 1
        self.token = self._refreshed_token
        self.valid = self._refreshed_token is not None


class FakeCatalogueBuilds:
    """A served catalogue whose build id is `build_id` and whose swaps `swap` drives."""

    def __init__(self, build_id: str = "build-a") -> None:
        self.build_id = build_id
        self._callbacks: list[Callable[[], None]] = []

    def request_build_id(self) -> str:
        return self.build_id

    def on_swap(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def swap(self, build_id: str) -> None:
        self.build_id = build_id
        for callback in self._callbacks:
            callback()
