"""A unit-unprobed survivor carries `observations_nothing_to_probe` in its preview.
The test attaches `_UNIT_CHECK_NOT_PROBED` through `_attach_observations`.
"""

from __future__ import annotations

from search.engine.region.candidates import _UNIT_CHECK_NOT_PROBED
from search.engine.region.codes import OBSERVATIONS_NOTHING_TO_PROBE
from search.engine.region.observations.dispatch import _attach_observations
from search.models import Spec, Warning
from tests.conftest import minimal_spec


def test_not_probed_survivor_observations_carry_nothing_to_probe():
    """An unprobed survivor gets omitted observations with reason `observations_nothing_to_probe`."""
    spec = Spec.model_validate(minimal_spec())
    warnings: list[Warning] = []

    attached = _attach_observations(spec, _UNIT_CHECK_NOT_PROBED, warnings)

    observations = attached.resolved.observations
    assert observations is not None, (
        "_UNIT_CHECK_NOT_PROBED must attach a real SpecObservations, so that a "
        "survivor no probe ever reached still carries a reason for its empty preview"
    )
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NOTHING_TO_PROBE
    assert observations.total_obs == 0
    assert observations.series == []
    assert warnings == []
