"""The session release with another code index, served to a surface or bound to a block, so
every reader of the bound release's index (`code_index.shared.get_index`,
`code_index.shared.dataset_of`) reads that index, as a production request reads its release's.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace

from code_index.query import CodeIndex
from graph_release.context import bound_release, current_release
from graph_release.service import Release, get_release_service


def release_with(code_index: CodeIndex) -> Release:
    """The bound release with `code_index` as its code index, opened by no service, so it
    reads as served."""
    release = current_release()
    return replace(release, views=replace(release.views, code_index=code_index), service=None)


class IndexServingRelease:
    """A release service serving `release_with(code_index)` on every request. Swap guards and
    callbacks register on the session's service, which never swaps."""

    def __init__(self, code_index: CodeIndex) -> None:
        self._service = get_release_service()
        self._release = release_with(code_index)

    def snapshot(self) -> Release:
        return self._release

    def serves(self, release: Release) -> bool:
        return release is self._release

    def register_swap_guard(
        self, guard: Callable[[], AbstractContextManager[object]]
    ) -> Callable[[], None]:
        return self._service.register_swap_guard(guard)

    def on_swap(self, callback: Callable[[Release], None]) -> Callable[[], None]:
        return self._service.on_swap(callback)


@contextmanager
def bound_index(code_index: CodeIndex) -> Iterator[CodeIndex]:
    """Binds `release_with(code_index)` for the `with` block."""
    with bound_release(release_with(code_index)):
        yield code_index
