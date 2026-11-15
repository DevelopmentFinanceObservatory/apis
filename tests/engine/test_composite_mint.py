"""`build_spec` minting a composite (`scope.composite.CompositeMint`) builds a Spec that states
the composite's identity and spec_id and counts the composite's components.

Each case runs a part and records the `build_spec` calls that minted a Spec as the composite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from code_index.composites import component_identities, table_identity
from code_index.shared import get_index
from search.engine import same_shape, spec_build
from search.engine.region import (
    candidate_grounding,
    grounding,
    phase_answer,
    phase_shared,
    promote_ground,
)
from search.engine.region.codes import ROLE_SCOPE_DEFAULTED
from search.engine.retrieval.bound_values import served_members
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_scope import DiscoveredScope
from search.engine.spec_id import identity_spec_id
from search.models import BaseIdentity, CoverageBare, Spec, Warning
from search.scope import composite
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._harness import offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _ScriptedLLM
from tests.engine.test_dac2a_destination_populations import (
    _HELD,
    _SCENARIOS,
    _VARIABLE,
    _graph_with_parties,
    _patched_coverage,
    _sdmx,
)
from tests.engine.test_promote_shape_sample import _USA, _YEARS

_CALLERS = (phase_answer, promote_ground, candidate_grounding, grounding, phase_shared, same_shape)
"""Every module that builds a Spec."""


@dataclass(frozen=True)
class _Built:
    """One `build_spec` call and the Spec it built."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    spec: Spec


def _recorded_builds(monkeypatch: pytest.MonkeyPatch) -> list[_Built]:
    """Every `build_spec` call any caller makes from here on, in order."""
    built: list[_Built] = []

    def recording(*args: Any, **kwargs: Any) -> Spec:
        spec = spec_build.build_spec(*args, **kwargs)
        built.append(_Built(args, kwargs, spec))
        return spec

    for module in _CALLERS:
        monkeypatch.setattr(module, "build_spec", recording)
    return built


def _minted(monkeypatch: pytest.MonkeyPatch, run: Callable[[], object]) -> list[_Built]:
    """Each `build_spec` call `run` makes that mints a Spec as a composite."""
    built = _recorded_builds(monkeypatch)
    run()
    return [item for item in built if item.kwargs.get("composite") is not None]


def _headline_coverage(document: dict[str, Any], *, origin: str = _USA) -> FakeCoverageService:
    """Series over 2005 to 2024 from `origin`, the United States unless named, for every
    member a mint of `document`'s table or of any of its components serves."""
    index = get_index()
    identities = [
        table_identity(index, document).identity,
        *(c.identity for c in component_identities(index, document) or ()),
    ]
    members = sorted(
        {m for doc in identities for m in served_members(BaseIdentity.model_validate(doc)) or ()}
    )
    roles = dict.fromkeys(members, RoleNames(donor="financingProvider", recipient=None))
    rows = [(dcid, "facet", origin, "", _YEARS) for dcid in members]
    return FakeCoverageService(build_coverage_index(rows, role_names=roles))


@pytest.fixture
def _dac2a_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DAC2a fixture's coverage snapshot (`test_dac2a_destination_populations`)."""
    _patched_coverage(monkeypatch, _HELD)


def _fresh(scenario: str) -> Callable[[], object]:
    return lambda: offline_resolve(
        _SCENARIOS[scenario](),
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(),
    )


@pytest.mark.parametrize(
    ("run", "composites"),
    [
        *(pytest.param(_fresh(name), 1, id=name) for name in ("H1", "H2", "H6")),
        # All recipients spans both sides, so the part reads total ODA and mints no composite.
        pytest.param(_fresh("H5"), 0, id="H5"),
    ],
)
@pytest.mark.usefixtures("_dac2a_coverage")
def test_a_minted_dac2a_composite_states_the_composite_s_identity_and_components(
    run: Callable[[], object], composites: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    minted = _minted(monkeypatch, run)

    assert len(minted) >= composites
    assert bool(minted) == bool(composites)
    for each in minted:
        mint = each.kwargs["composite"]
        assert each.spec.spec_id == mint.spec_id == identity_spec_id(mint.identity)
        assert each.spec.definition.identity == mint.identity
        assert len(mint.components()) == len(mint.served)


@pytest.mark.parametrize("listed", [True, False], ids=["listed", "unlisted"])
@pytest.mark.usefixtures("_dac2a_coverage")
def test_the_mint_states_the_defaulted_scope_count_its_role_axis_fact_counts(
    listed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `role_scope_defaulted` warning on the composite's open axis, keyed to the minted
    Spec or to none, states what the fresh answer's response states there, the count the
    Spec's role axis fact states, found by its code and axis whatever its text says, also
    where the snapshot cannot list the composite's parties and the fact keeps the table's
    count. A warning on an axis the composite counts nothing on stands, as does one keyed
    to another reading beside the answer."""
    if not listed:
        monkeypatch.setattr(composite, "_lists_parties", lambda *_args: False)
    responses: list[Any] = []
    (first, *_rest) = _minted(monkeypatch, lambda: responses.append(_fresh("H2")()))
    (part,) = responses[0].parts
    (answered,) = [w for w in part.warnings if w.code == ROLE_SCOPE_DEFAULTED]
    minted = first.spec
    mint = first.kwargs["composite"]
    on_axis = Warning(
        code=ROLE_SCOPE_DEFAULTED,
        severity="info",
        message="anything",
        params={"axis": "destination"},
    )
    keyed = on_axis.model_copy(update={"spec_id": minted.spec_id})
    off_axis = on_axis.model_copy(update={"params": {"axis": "origin"}})
    other = Warning(code="observations_scope_partial", severity="warn", message="partial")
    elsewhere = on_axis.model_copy(update={"spec_id": "spec_another_reading"})

    stated = mint.role_scope_warnings((on_axis, keyed, off_axis, other, elsewhere), minted)

    assert stated[0].message == answered.message
    assert stated[1].message == stated[0].message
    assert stated[2:] == (off_axis, other, elsewhere)


@pytest.mark.usefixtures("_dac2a_coverage")
def test_a_composite_candidate_listing_no_member_drops_the_defaulted_scope_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounding a candidate minted as a composite whose Spec lists no member on the open
    axis removes the `role_scope_defaulted` warning, as it does for any candidate
    (`grounding.restate_role_scope_warnings`)."""
    (first, *_rest) = _minted(monkeypatch, _fresh("H2"))
    mat = Materialised(
        sv_dcids=[],
        has_data=True,
        coverage=CoverageBare(has_data=True),
        discovered_scope=DiscoveredScope(
            axis="destination", role_name="recipient", member_dcids=[], filters={}
        ),
    )
    warning = Warning(
        code=ROLE_SCOPE_DEFAULTED,
        severity="info",
        message="anything",
        params={"axis": "destination"},
    )
    state = SimpleNamespace(
        opposite_labels=lambda dcids: list(dcids),
        mat_result=mat,
        place=SimpleNamespace(roles=SimpleNamespace(roles={}, unnamed_wholes=frozenset())),
        warnings=[warning],
    )
    entry = SimpleNamespace(
        candidate=SimpleNamespace(mat=mat),
        scope_members=(),
        minted=SimpleNamespace(mint=first.kwargs["composite"]),
        spec=first.spec,
    )

    candidate_grounding.restate_scope_warning(cast(Any, state), [cast(Any, entry)])

    assert state.warnings == []
