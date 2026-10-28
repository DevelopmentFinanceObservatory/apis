"""Background mint admission.

A mint posted with `X-DFO-Mint-Admission: background` (`BACKGROUND_HEADER`) warms data no
caller waits on yet, so it is admitted only without waiting and only while it leaves a place for
the callers that do. Where a foreground mint would wait or be refused, it is refused at once with
`MintBackgroundDeclined` and nothing queues:

- its work slot (`door_executor.DoorExecutor.run`) and materialise permit
  (`admission.bounded_materialise_slot`), held for the whole fetch, are taken only while at
  least `BACKGROUND_HEADROOM` of each stays free after, and a release replaced under it refuses;
- a rebuild of the subject another caller is running (`admission.RebuildFlights.turn`) refuses
  it, since that rebuild already warms the subject and waiting would hold a work slot;
- an over-budget mint records a new operation only while `BACKGROUND_HEADROOM` places of the
  operation backlog stay free after (`OperationStore.create_or_get`). The operation runs on the
  worker as any other, since its id is shared by every caller of the same rendering.

Once admitted, it runs as a foreground mint does. It takes the groupby fold permit
(`render._fold_permit`) as a foreground mint does, waiting up to its deadline, since by then it
holds a fetched frame that callers who joined its rebuild wait on. A head that already links the
rendering answers it with no permit.

The door binds the headroom to the request's context (`leaving_headroom`), which the work
executor carries into the mint's thread, and each limit reads it (`admission_headroom`). A
request without the header binds none and takes every limit as a foreground mint does.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

BACKGROUND_HEADER = "X-DFO-Mint-Admission"
BACKGROUND_VALUE = "background"

# The work slots, materialise permits and backlog places a background mint leaves free for others.
BACKGROUND_HEADROOM = 1

_headroom: ContextVar[int] = ContextVar("datasets_admission_headroom", default=0)


class HeadroomSemaphore(threading.BoundedSemaphore):
    """A bounded semaphore that can also take a unit only while others stay free."""

    # Set by `threading.Semaphore.__init__`, which guards `_value` with it.
    _cond: threading.Condition

    def acquire_leaving(self, headroom: int) -> bool:
        """Take one unit without waiting if more than `headroom` units are free, and return
        whether it was taken. A `headroom` of zero is a non-blocking `acquire`."""
        with self._cond:
            if self._value <= headroom:
                return False
            self._value -= 1
            return True


def admission_headroom() -> int:
    """The work slots, materialise permits and operation backlog places the current request must
    leave free. It is zero unless the request is a background mint (`leaving_headroom`)."""
    return _headroom.get()


@contextmanager
def leaving_headroom(headroom: int) -> Iterator[None]:
    """Bind `headroom` as the units of each limit the work started in the block leaves free."""
    token = _headroom.set(headroom)
    try:
        yield
    finally:
        _headroom.reset(token)


__all__ = [
    "BACKGROUND_HEADER",
    "BACKGROUND_HEADROOM",
    "BACKGROUND_VALUE",
    "HeadroomSemaphore",
    "admission_headroom",
    "leaving_headroom",
]
