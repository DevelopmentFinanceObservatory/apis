"""Typed access to lifespan-owned values stored on the composed FastAPI app.

Each surface sets its slot after building dependencies and clears it on every lifespan exit.
Distinct `name` values prevent surfaces on one app from colliding. `unset_exception` (a type
buildable with no arguments) can provide the surface's classified refusal when a request
arrives before the slot is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from fastapi import FastAPI, Request

# FastAPI resolves dependency annotations with `get_type_hints`, so `T` must be a module global.
T = TypeVar("T")

_UNSET = object()


class StateNotSet(Exception):
    """Raised when a request reads a slot before its surface sets it during lifespan startup."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"app.state has no {name!r} -- the owning surface's lifespan has not "
            "set it yet, or never runs on this composed app"
        )


@dataclass(frozen=True, slots=True)
class StateSlot(Generic[T]):  # noqa: UP046 (see the T TypeVar comment above)
    """One named attribute on `app.state`, typed as `T` for its caller."""

    name: str
    unset_exception: type[BaseException] | None = None

    def set(self, app: FastAPI, value: T) -> None:
        setattr(app.state, self.name, value)

    def clear(self, app: FastAPI) -> None:
        if hasattr(app.state, self.name):
            delattr(app.state, self.name)

    def __call__(self, request: Request) -> T:
        value = getattr(request.app.state, self.name, _UNSET)
        if value is _UNSET:
            if self.unset_exception is not None:
                raise self.unset_exception()
            raise StateNotSet(self.name)
        # `set` is the only writer for this key and stores `T`; the cast narrows `getattr`'s result.
        return cast(T, value)
