"""One value per event loop, for the clients whose pools and semaphores belong to the loop
that made them: the Gemini client's connection pools (`clients.llm`) and Jev's connection
pools and request semaphores (`binders.jev_client`)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable


class LoopLocal[T]:
    """A `T` per event loop. Values of closed loops are dropped on the next lookup. The
    lock serves loops on different threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[asyncio.AbstractEventLoop, T] = {}

    def get(self, build: Callable[[], T]) -> T:
        """The running loop's value, built by `build` on the loop's first lookup."""
        loop = asyncio.get_running_loop()
        with self._lock:
            for closed in [known for known in self._values if known.is_closed()]:
                del self._values[closed]
            value = self._values.get(loop)
            if value is None:
                value = self._values[loop] = build()
            return value

    def pop(self) -> T | None:
        """Remove and return the running loop's value, None when it has none."""
        with self._lock:
            return self._values.pop(asyncio.get_running_loop(), None)
