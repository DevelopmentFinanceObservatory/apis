"""Share one 5-second drain deadline across a surface's executors.

The Datasets surface drains its work and read executors one after the other
(`datasets_api.deps.open_deps`). One shared deadline caps both drains at 5
seconds together. Presentation's doors use it too.
"""

from __future__ import annotations

import time

# The whole surface's own shutdown drain budget, in seconds.
SHUTDOWN_BUDGET_SECONDS = 5.0


class ShutdownBudget:
    """A monotonic deadline shared by every drain using this instance.
    The first `remaining()` call arms it, when shutdown draining begins.
    """

    def __init__(self, *, total_seconds: float = SHUTDOWN_BUDGET_SECONDS) -> None:
        self._total_seconds = total_seconds
        self._deadline: float | None = None

    def remaining(self) -> float:
        """Seconds left until the deadline, clamped at zero. The first call arms the deadline."""
        if self._deadline is None:
            self._deadline = time.monotonic() + self._total_seconds
        return max(0.0, self._deadline - time.monotonic())
