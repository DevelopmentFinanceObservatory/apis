"""Build eval requests and Langfuse item ids from cases.

`resubmit_identity` builds a dataset_resubmit case's identity, which the runner posts.
`item_id_for_case` names a case's Langfuse dataset item.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from search.engine.identity_edit import synthesize_base
from search.eval.cases import Case, Resubmit
from search.models import (
    BaseIdentity,
    BindingValue,
    IdentityFacetBinding,
    Slot,
    TimeWindow,
)


def _requested_window(slots: Iterable[Slot]) -> TimeWindow | None:
    """Return the window a bound ``when`` slot carries, or None when it names none.

    Identity hashing excludes the ``when`` axis; the window rides on ``requested_window``.
    """
    for slot in slots:
        binding = slot.binding
        if (
            slot.key.axis == "when"
            and isinstance(binding, BindingValue)
            and binding.value.value_kind == "time_window"
        ):
            return binding.value.time_window
    return None


def resubmit_identity(resubmit: Resubmit) -> tuple[str | None, BaseIdentity]:
    """Synthesize a dataset_resubmit case's base spec id and identity.

    A bound ``when`` slot's window becomes the identity's ``requested_window``. Raises
    ``ValueError`` when the payload names no entity or StatVar to anchor an identity.
    """
    return _base_identity(
        resubmit.shape.model_dump(mode="json", exclude_unset=True),
        resubmit.slots,
        entity_dcids=resubmit.entity_dcids,
        origin_dcids=resubmit.origin_dcids,
        stat_var_dcids=resubmit.stat_var_dcids,
        facets=resubmit.facets,
    )


def _base_identity(
    shape: Mapping[str, object],
    slots: list[Slot],
    *,
    entity_dcids: list[str] | None,
    origin_dcids: list[str] | None,
    stat_var_dcids: list[str] | None,
    facets: Mapping[str, IdentityFacetBinding] | None,
) -> tuple[str | None, BaseIdentity]:
    base_spec_id, identity = synthesize_base(
        shape,
        slots,
        entity_dcids=entity_dcids,
        origin_dcids=origin_dcids,
        resolved_stat_var_dcids=stat_var_dcids,
        requested_window=_requested_window(slots),
        facets=facets,
    )
    if identity is None:
        raise ValueError("dataset_resubmit needs an identity definition")
    return base_spec_id, identity


def item_id_for_case(case: Case, dataset_name: str) -> str:
    """Return a case's item id, scoped by dataset name because Langfuse ids are global."""
    return f"{dataset_name}:search-case-{case.id}"
