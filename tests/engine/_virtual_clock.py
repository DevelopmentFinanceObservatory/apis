"""Run a resolve on one thread under a virtual clock and record each request's span in ticks.

`VirtualRequests` measures how the requests of one resolve overlap without reading a wall
clock, so machine load cannot change the result. `install` puts three pieces in place:

- The offline doubles' public methods: each outermost request takes one tick and records
  its name, first argument, and start and end ticks. A method named in `free` is a call the
  live client answers without a request, and takes no tick.
- The event loop's `run_in_executor`, which `asyncio.to_thread` calls: the function runs
  while the loop waits for it, at the virtual time it is submitted, its requests follow one
  another from that time, and its future resolves at the tick the last of them ends.
  Functions launched together therefore overlap, and a function awaited after another
  follows it. A function for the default executor runs on the loop's thread; one for a
  named executor runs on that executor, which may be bound to its own thread.
- `ThreadPoolExecutor` in the modules given: each job runs on the calling thread from the
  tick it is submitted at, so the jobs of one pool overlap, and joining a job (`result`,
  `shutdown`) moves the clock to the tick it ends. A pool a module builds at import is
  replaced by an executor that runs its jobs the same way. `install` always replaces the
  pool that settles a StatVar's default place scope (`materialise.place_scope`), so its
  availability reads run on the clock, beside the request that waits for them.

`VirtualRequests.run` runs a coroutine on `VirtualClockLoop`, whose time moves only when a
timer is due or a request takes its tick. A request from a thread the loop is not waiting
for raises, and so does a loop left waiting on such a thread, since either runs off the
virtual clock.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import selectors
import threading
from collections.abc import Callable, Coroutine, Iterable, Iterator
from concurrent.futures import Future
from types import ModuleType
from typing import Any

import pytest

from materialise import place_scope

# One request's duration in loop seconds: a power of two, so sums of ticks are exact.
_TICK = 2.0**-10


class _TimerSkippingSelector:
    """Wraps a selector: a wait for the next timer advances the loop's clock to it instead
    of sleeping. A wait with no timer due would never end, so it raises."""

    def __init__(self, inner: selectors.BaseSelector, loop: VirtualClockLoop) -> None:
        self._inner = inner
        self._loop = loop

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        if timeout is None:
            raise AssertionError("the loop waits on work that runs off the virtual clock")
        if timeout > 0:
            self._loop.advance(timeout)
        return self._inner.select(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class VirtualClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose time is a counter. It moves when the loop would wait for a timer,
    and `clock` moves it by hand. Executor functions run while the loop waits for them
    (module docstring)."""

    def __init__(self) -> None:
        self._now = 0.0
        # The clock of the executor function running now, else `None`.
        self._function_clock: float | None = None
        # Whether that function runs on a named executor's thread.
        self.waiting_for_named_executor = False
        super().__init__(_TimerSkippingSelector(selectors.DefaultSelector(), self))

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    @property
    def clock(self) -> float:
        """The virtual time of the code running now: the executor function's when one runs,
        else the loop's."""
        return self._now if self._function_clock is None else self._function_clock

    @clock.setter
    def clock(self, value: float) -> None:
        if self._function_clock is None:
            self._now = value
        else:
            self._function_clock = value

    def run_in_executor(self, executor: Any, func: Callable[..., Any], *args: Any) -> Any:
        if self._function_clock is not None:
            raise AssertionError("an executor function submits another through the loop")
        future = self.create_future()
        self._function_clock = self._now
        self.waiting_for_named_executor = executor is not None
        try:
            outcome = func(*args) if executor is None else executor.submit(func, *args).result()
        except BaseException as exc:
            settle = functools.partial(_settle, future, error=exc)
        else:
            settle = functools.partial(_settle, future, outcome)
        finally:
            end, self._function_clock = self._function_clock, None
            self.waiting_for_named_executor = False
        self.call_at(end, settle)
        return future


def _settle(
    future: asyncio.Future, outcome: Any = None, *, error: BaseException | None = None
) -> None:
    if future.cancelled():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(outcome)


class _JoinedFuture(Future):
    """A finished job of an `_InlineExecutor`; reading it moves the clock to its end."""

    def __init__(self, requests: VirtualRequests) -> None:
        super().__init__()
        self._requests = requests
        self.end = 0.0

    def result(self, timeout: float | None = None) -> Any:
        self._requests.join(self.end)
        return super().result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._requests.join(self.end)
        return super().exception(timeout)


class VirtualRequests:
    """Records the span of every request of one resolve in ticks of a virtual clock."""

    def __init__(self) -> None:
        # (name, first argument, start tick, end tick) of each outermost request.
        self.spans: list[tuple[str, Any, int, int]] = []
        self._loop = VirtualClockLoop()
        self._thread = threading.current_thread()
        self._nested = 0

    @property
    def clock(self) -> float:
        return self._loop.clock

    @clock.setter
    def clock(self, value: float) -> None:
        self._loop.clock = value

    def join(self, end: float) -> None:
        """Move the clock to `end` when it is behind it."""
        self.clock = max(self.clock, end)

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Run `coro` to completion on the virtual-clock loop."""
        with asyncio.Runner(loop_factory=lambda: self._loop) as runner:
            return runner.run(coro)

    def install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        doubles: Iterable[type],
        pools: Iterable[ModuleType],
        executors: Iterable[tuple[ModuleType, str]] = (),
        free: Iterable[str] = (),
    ) -> None:
        """Patch the public methods of `doubles` but those named in `free`, the
        `ThreadPoolExecutor` name of each module in `pools`, and each `(module, name)` in
        `executors`, a pool the module built at import. The default place scope's settle pool
        is replaced in every case."""
        free = frozenset(free)
        monkeypatch.setattr(place_scope, "_pool", self._executor_type()())
        for owner in doubles:
            for name, method in list(vars(owner).items()):
                if not name.startswith("_") and name not in free and inspect.isfunction(method):
                    monkeypatch.setattr(owner, name, self._request(name, method))
        for module in pools:
            monkeypatch.setattr(module, "ThreadPoolExecutor", self._executor_type())
        for module, name in executors:
            monkeypatch.setattr(module, name, self._executor_type()())

    def first(self, name: str, argument: Any = None) -> tuple[int, int]:
        """The span of the first request named `name`, with `argument` first when given."""
        return min(
            (start, end)
            for n, a, start, end in self.spans
            if n == name and (argument is None or a == argument)
        )

    def serial_depth(self) -> int:
        """The longest chain of requests each starting once the previous one has ended."""
        spans = sorted((start, end) for _, _, start, end in self.spans)
        chain: list[int] = []
        for index, (start, _end) in enumerate(spans):
            before = [chain[j] for j, (_s, end) in enumerate(spans[:index]) if start >= end]
            chain.append(1 + max(before, default=0))
        return max(chain, default=0)

    def _request(self, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def request(double: object, *args: Any, **kwargs: Any) -> Any:
            on_clock = (
                threading.current_thread() is self._thread or self._loop.waiting_for_named_executor
            )
            if not on_clock:
                raise AssertionError(f"{name} runs on a thread off the virtual clock")
            if self._nested:
                return method(double, *args, **kwargs)
            start = self.clock
            self.clock = start + _TICK
            self._nested += 1
            try:
                return method(double, *args, **kwargs)
            finally:
                self._nested -= 1
                tick = round(start / _TICK)
                self.spans.append((name, args[0] if args else None, tick, tick + 1))

        return request

    def _executor_type(self) -> type:
        requests = self

        class _InlineExecutor:
            """Stands in for `ThreadPoolExecutor` (module docstring)."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._ends: list[float] = []

            def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
                future = _JoinedFuture(requests)
                start = requests.clock
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
                future.end = requests.clock
                self._ends.append(future.end)
                requests.clock = start
                return future

            def map(self, fn: Callable[..., Any], *iterables: Iterable[Any]) -> Iterator[Any]:
                futures = [self.submit(fn, *args) for args in zip(*iterables, strict=False)]
                return iter([future.result() for future in futures])

            def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
                if wait:
                    requests.join(max(self._ends, default=requests.clock))

            def __enter__(self) -> _InlineExecutor:
                return self

            def __exit__(self, *exc_info: object) -> None:
                self.shutdown(wait=True)

        return _InlineExecutor
