"""Option ids: `<axis>:<value>`, one string per option on every part.

Sector options name their CRS purpose codes by number (`sector:120+130`); the other axes
name a declared value (`region:members_and_residuals`, `measure_basis:gross`,
`provider:inclusive`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import get_args

from code_index.scope_conventions import MeasureBasis
from search.models import SCOPE_AXES, ProviderScope, ScopeAxis

REGION_MEMBERS_AND_RESIDUALS = "region:members_and_residuals"
"""A summed region: its member countries and its regional residual buckets."""
REGION_MEMBERS_ONLY = "region:members_only"
"""A region read as its member countries alone."""

_CODE_NUMBER = re.compile(r"(\d+)$")


def option_id(axis: ScopeAxis, value: str) -> str:
    """The id of `axis`'s option `value`."""
    return f"{axis}:{value}"


def option_axis(option: str) -> ScopeAxis:
    """The axis an option id names. Raises `ValueError` for an id naming no axis."""
    axis = option.partition(":")[0]
    if axis not in SCOPE_AXES:
        raise ValueError(f"option id {option!r} names no scope axis")
    return axis


def sector_option_id(codes: Iterable[str]) -> str:
    """The sector option binding `codes`, purpose-code dcids ending in their number, in
    ascending number order: `sector:120+130`. Raises `ValueError` for a dcid with no number
    or an empty list."""
    numbers = []
    for code in codes:
        if (match := _CODE_NUMBER.search(code)) is None:
            raise ValueError(f"sector code {code!r} ends in no code number")
        numbers.append(match.group(1))
    if not numbers:
        raise ValueError("a sector option binds at least one code")
    return option_id("sector", "+".join(sorted(set(numbers), key=int)))


def measure_basis_option_id(basis: MeasureBasis) -> str:
    """The measure-basis option `basis`: `measure_basis:gross`."""
    return option_id("measure_basis", basis)


def provider_option_id(scope: ProviderScope) -> str:
    """The provider option `scope`: `provider:inclusive`."""
    return option_id("provider", scope)


MEASURE_BASIS_OPTIONS: tuple[str, ...] = tuple(
    measure_basis_option_id(basis) for basis in get_args(MeasureBasis)
)
"""Every measure-basis option id."""
PROVIDER_OPTIONS: tuple[str, ...] = tuple(
    provider_option_id(scope) for scope in get_args(ProviderScope)
)
"""Every provider option id."""
REGION_OPTIONS: tuple[str, ...] = (REGION_MEMBERS_AND_RESIDUALS, REGION_MEMBERS_ONLY)
"""Every region option id."""
