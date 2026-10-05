"""Tests the origin catalogue's manifest warnings and the `excluded_by_registry` and
UN refusals."""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping, Sequence

import pytest

from datasets_api.codes_materialise import ExcludedByRegistry, UnSystemUnavailable
from datasets_api.doors.definition import (
    DefinitionMaterialiseRefused,
    classify_definition_exception,
)
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.origin import (
    WITHHELD_READ_MAX_BYTES,
    WithheldReader,
    origin_reads,
    un_unavailable_warnings,
    withheld_warning,
    withheld_warnings,
)
from dc_graph import GraphCapExceededError, GraphDeadlineExceededError
from dc_sdmx import ContainedIn, SdmxCapExceededError, SdmxTransportError
from graph_generation import FakeGeneration, GenerationStamp
from materialise import MaterialisationRequest
from materialise.errors import DefaultFacetUndefinedError
from materialise.place_scope import PlaceScopeReadError
from materialise.window import Coverage, RequestedWindow
from origin_catalogue import UnReadFailure, note_un_blocked_places
from origin_catalogue.context import un_unavailability_record
from origin_catalogue.graph import build_graph_client
from tests.materialise._scope_availability import ScopeAvailability

_KEN, _NGA, _UGA = "country/KEN", "country/NGA", "country/UGA"
_UN_VARIABLE = "undata/who/MORT_U5_RATE"


class _WithheldGraph:
    """Answers the unfiltered re-read with the given provenances at every requested entity,
    or at `at` alone when given, and records each call with its deadline and byte cap."""

    def __init__(
        self,
        withheld: Sequence[str] = (),
        *,
        at: Sequence[str] | None = None,
        fails: Exception | None = None,
    ) -> None:
        self._withheld = tuple(withheld)
        self._at = None if at is None else set(at)
        self._fails = fails
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.limits: list[tuple[float | None, int | None]] = []

    def withheld_at(
        self,
        variables: Sequence[str],
        entities: Sequence[str],
        *,
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> Mapping[str, tuple[str, ...]]:
        self.calls.append((tuple(variables), tuple(entities)))
        self.limits.append((deadline, max_bytes))
        if self._fails is not None:
            raise self._fails
        if not self._withheld:
            return {}
        return {
            entity: self._withheld for entity in entities if self._at is None or entity in self._at
        }


_GENERATIONS = itertools.count()


class _ScopedWithheldGraph(_WithheldGraph):
    """A `_WithheldGraph` that also reaches a UN deployment whose availability lists
    `un_places` for every UN variable, or fails with `un_fails`, and records each such read.
    It holds every place it is asked about as a node. The countries fallback reads `countries`
    as Earth's countries and `held` as the `(StatVar, place)` pairs with admitted data. Each
    graph has a generation of its own, so no test reads another's memoised scope."""

    def __init__(
        self,
        *args,
        un_places: Sequence[str] = (),
        un_fails: Exception | None = None,
        countries: Sequence[str] = (),
        held: Sequence[tuple[str, str]] = (),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._un_places = list(un_places)
        self._un_fails = un_fails
        self._countries = list(countries)
        self._held = frozenset(held)
        self._generation = FakeGeneration(token=f"scoped-withheld-{next(_GENERATIONS)}")
        self.scope_reads: list[tuple[str, ...]] = []

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._generation.stamp()

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return {parent: list(self._countries) for parent in parent_dcids}

    def entities_with_data_by_variable(
        self,
        variable_dcids: Sequence[str],
        entity_dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        return {
            variable: [entity for entity in entity_dcids if (variable, entity) in self._held]
            for variable in variable_dcids
        }

    def un_availability(self) -> _ScopedWithheldGraph:
        return self

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: object = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        self.scope_reads.append(tuple(variables))
        if self._un_fails is not None:
            raise self._un_fails
        return list(self._un_places)

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return {dcid: dict.fromkeys(properties, ("Place",)) for dcid in dcids}

    def close(self) -> None:
        return None


def _scope(*held: tuple[str, str]) -> ScopeAvailability:
    """Availability for the default place scope: `held` lists the `(StatVar, place)` pairs with
    data."""
    return ScopeAvailability(data=held)


class _FailingAvailability:
    def __init__(self, failure: Exception) -> None:
        self._failure = failure

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(self, **_kwargs: object) -> list[str]:
        raise self._failure


def _unbound(stat_var: str = "dcid:Count_Person") -> MaterialisationRequest:
    """A subject-only request that binds no entity, so it covers the default place scope."""
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def _request(
    stat_var: str = "dcid:Count_Person",
    entities: object = (_KEN, _NGA),
    roles: tuple[str, ...] = ("observationAbout",),
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={roles[0]: entities},  # ty: ignore[invalid-argument-type]
        declared_roles=roles,
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def _undefined(request: MaterialisationRequest | None) -> DefaultFacetUndefinedError:
    return DefaultFacetUndefinedError("no candidate", request=request)


def _served(*entities: str) -> Coverage:
    return Coverage(
        rows=len(entities),
        dimensions=(),
        window=None,
        entity_refs={"observationAbout": frozenset(entities)},
    )


def test_an_unserved_entity_with_withheld_data_names_the_provenances():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    found = withheld_warning(graph, [(_request(), _served(_KEN))], deadline=None)

    assert found is not None
    index, warning = found
    assert index == 0
    assert warning.code == "withheld_by_registry"
    assert warning.severity == "warn"
    assert "country/NGA" in warning.message
    assert "dc/base/UN_SDG" in warning.message
    assert graph.calls == [(("Count_Person",), (_NGA,))]


def test_every_entity_served_reads_nothing():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    assert withheld_warnings(graph, [(_request(), _served(_KEN, _NGA))], deadline=None) == ()
    assert graph.calls == []


def test_an_unserved_entity_with_nothing_withheld_warns_nothing():
    graph = _WithheldGraph()

    assert withheld_warnings(graph, [(_request(), _served())], deadline=None) == ()
    assert graph.calls == [(("Count_Person",), (_KEN, _NGA))]


def test_the_re_read_runs_once_for_the_first_request_with_an_unserved_entity():
    graph = _WithheldGraph()
    served = [
        (_request("dcid:A"), _served(_KEN, _NGA)),
        (_request("dcid:B"), _served(_KEN)),
        (_request("dcid:C", entities=(_UGA,)), _served()),
    ]

    assert withheld_warning(graph, served, deadline=None) is None
    assert graph.calls == [(("B",), (_NGA,))]


def test_the_warning_carries_the_index_of_its_request():
    graph = _WithheldGraph(withheld=["dc/base/X"])
    served = [(_request("dcid:A"), _served(_KEN, _NGA)), (_request("dcid:B"), _served(_KEN))]

    found = withheld_warning(graph, served, deadline=None)

    assert found is not None and found[0] == 1


def test_role_bearing_and_scoped_requests_are_skipped():
    graph = _WithheldGraph(withheld=["dc/base/X"])
    served = [
        (_request(roles=("provider", "recipient")), _served()),
        (_request(entities=ContainedIn(parent="Earth", child_type="Country")), _served()),
    ]

    assert withheld_warnings(graph, served, deadline=None) == ()
    assert graph.calls == []


def test_a_request_binding_no_entity_is_checked_at_the_countries_it_left_unserved():
    graph = _ScopedWithheldGraph(withheld=["dc/base/UN_SDG"])

    found = withheld_warning(
        graph, [(_unbound(), _served(_KEN))], default_scope=(_KEN, _NGA, _UGA), deadline=5.0
    )

    assert found is not None
    index, warning = found
    assert index == 0
    assert f"served no observations for {_NGA}, {_UGA};" in warning.message
    # Earth is the scope's fallback and no part of a country answer, so it is not re-read.
    assert graph.calls == [(("Count_Person",), (_NGA, _UGA))]
    # The countries are the ones the result carries, so the warning reads no containment.
    assert graph.scope_reads == []


def test_a_request_binding_no_entity_is_checked_at_each_default_place_it_did_not_serve():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    (warning,) = withheld_warnings(
        graph, [(_unbound(), _served(_UGA))], default_scope=(_KEN, _NGA, _UGA), deadline=None
    )

    assert f"served no observations for {_KEN}, {_NGA};" in warning.message


def test_a_request_binding_no_entity_served_at_every_country_reads_nothing():
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])

    assert (
        withheld_warnings(
            graph, [(_unbound(), _served(_KEN, _NGA))], default_scope=(_KEN, _NGA), deadline=None
        )
        == ()
    )
    assert graph.calls == []


def test_a_request_binding_no_entity_is_skipped_without_a_default_scope():
    graph = _ScopedWithheldGraph(withheld=["dc/base/X"])

    assert withheld_warnings(graph, [(_unbound(), _served())], deadline=1.0) == ()
    assert graph.calls == []
    assert graph.scope_reads == []


def test_the_warning_names_only_the_entities_the_re_read_reports():
    # NGA has admitted data outside the served window, so the re-read leaves it out.
    graph = _WithheldGraph(withheld=["dc/base/UN_SDG"], at=[_UGA])
    request = _request(entities=(_KEN, _NGA, _UGA))

    (warning,) = withheld_warnings(graph, [(request, _served(_KEN))], deadline=None)

    assert f"served no observations for {_UGA};" in warning.message
    assert _NGA not in warning.message


def test_the_re_read_runs_under_the_deadline_and_the_byte_cap():
    graph = _WithheldGraph(withheld=["dc/base/X"])

    withheld_warnings(graph, [(_request(), _served())], deadline=123.0)

    assert graph.limits == [(123.0, WITHHELD_READ_MAX_BYTES)]


@pytest.mark.parametrize(
    "failure",
    [
        GraphDeadlineExceededError("deadline"),
        GraphCapExceededError("cap", cap_kind="bytes", cap=1, observed=2),
    ],
)
def test_a_failed_re_read_leaves_the_answer_without_the_warning(failure: Exception):
    graph = _WithheldGraph(withheld=["dc/base/X"], fails=failure)

    assert withheld_warnings(graph, [(_request(), _served())], deadline=1.0) == ()


def test_a_graph_without_the_re_read_reads_nothing():
    assert withheld_warnings(object(), [(_request(), _served())], deadline=None) == ()


class TestOriginReads:
    def test_an_empty_probe_under_withheld_provenances_refuses_excluded_by_registry(self):
        graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])
        failing = _request("dcid:B", entities=(_NGA, _UGA))

        with (
            pytest.raises(ExcludedByRegistry) as refused,
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            raise _undefined(failing)

        assert refused.value.code == "excluded_by_registry"
        assert refused.value.status == 422
        assert refused.value.variables == ("B",)
        assert refused.value.provenances == ("dc/base/UN_SDG",)
        assert "dc/base/UN_SDG" in str(refused.value)
        assert isinstance(refused.value.__cause__, DefaultFacetUndefinedError)

    def test_the_re_read_covers_only_the_failing_member(self):
        graph = _WithheldGraph(withheld=["dc/base/UN_SDG"])
        failing = _request("dcid:B", entities=(_NGA, _UGA))

        with (
            pytest.raises(ExcludedByRegistry),
            origin_reads(graph, availability=_scope(), deadline=7.0),
        ):
            raise _undefined(failing)

        assert graph.calls == [(("B",), (_NGA, _UGA))]
        assert graph.limits == [(7.0, WITHHELD_READ_MAX_BYTES)]

    def test_a_refusal_naming_no_member_reads_nothing_and_passes_through(self):
        graph = _WithheldGraph(withheld=["dc/base/X"])

        with (
            pytest.raises(DefaultFacetUndefinedError),
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            raise _undefined(None)

        assert graph.calls == []

    def test_an_empty_probe_with_nothing_withheld_keeps_default_facet_undefined(self):
        graph = _WithheldGraph()

        with (
            pytest.raises(DefaultFacetUndefinedError),
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            raise _undefined(_request())

        assert len(graph.calls) == 1

    def test_a_failed_re_read_keeps_default_facet_undefined(self):
        graph = _WithheldGraph(withheld=["dc/base/X"], fails=GraphDeadlineExceededError("late"))

        with (
            pytest.raises(DefaultFacetUndefinedError),
            origin_reads(graph, availability=_scope(), deadline=1.0),
        ):
            raise _undefined(_request())

    def test_a_role_bearing_request_reads_nothing_and_keeps_the_refusal(self):
        graph = _WithheldGraph(withheld=["dc/base/X"])

        with (
            pytest.raises(DefaultFacetUndefinedError),
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            raise _undefined(_request(roles=("provider", "recipient")))

        assert graph.calls == []

    def test_a_served_materialisation_reads_nothing(self):
        graph = _WithheldGraph(withheld=["dc/base/X"])

        with origin_reads(graph, availability=_scope(), deadline=None):
            pass

        assert graph.calls == []

    def test_a_un_member_at_refused_places_refuses_upstream_error_naming_them(self):
        graph = _WithheldGraph(withheld=["undata/p/X"])
        request = _request(f"dcid:{_UN_VARIABLE}", entities=(_KEN, _NGA))

        with (
            pytest.raises(UnSystemUnavailable) as refused,
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            note_un_blocked_places([_NGA, _UGA])
            raise _undefined(request)

        assert refused.value.places == (_NGA,)
        assert (refused.value.code, refused.value.status, refused.value.retryable) == (
            "upstream_error",
            502,
            True,
        )
        assert _NGA in str(refused.value) and _UN_VARIABLE in str(refused.value)
        assert graph.calls == []

    def test_a_un_member_whose_read_failed_alongside_ours_refuses_upstream_error(self):
        request = _request(f"dcid:{_UN_VARIABLE}", entities=(_KEN, _NGA))

        with (
            pytest.raises(UnSystemUnavailable) as refused,
            origin_reads(_WithheldGraph(), availability=_scope(), deadline=None),
        ):
            record = un_unavailability_record()
            assert record is not None
            record.add_failure(
                UnReadFailure(variables=(_UN_VARIABLE,), places=(_KEN, _NGA, _UGA), detail="503")
            )
            raise _undefined(request)

        assert refused.value.places == (_KEN, _NGA)

    def test_a_un_member_binding_no_place_refuses_upstream_error_when_its_reads_failed(self):
        graph = _ScopedWithheldGraph(un_places=[_KEN, _NGA])
        request = _unbound(f"dcid:{_UN_VARIABLE}")

        with (
            pytest.raises(UnSystemUnavailable) as refused,
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            record = un_unavailability_record()
            assert record is not None
            record.add_failure(
                UnReadFailure(variables=(_UN_VARIABLE,), places=(_KEN, _NGA, "Earth"), detail="503")
            )
            raise _undefined(request)

        assert refused.value.places == (_KEN, _NGA, "Earth")
        assert (refused.value.code, refused.value.status, refused.value.retryable) == (
            "upstream_error",
            502,
            True,
        )
        assert graph.calls == []

    def test_a_un_member_binding_no_place_names_refused_places_in_its_no_data_refusal(self):
        # Every country is in its scope, so the places the UN front door refuses on every read
        # are too. A retry would meet the same refusal, so the refusal stays no data and names
        # them, as the warning on a served request would.
        graph = _ScopedWithheldGraph(un_places=[_KEN, _NGA])
        request = _unbound(f"dcid:{_UN_VARIABLE}")

        with (
            pytest.raises(DefaultFacetUndefinedError) as refused,
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            note_un_blocked_places([_NGA])
            raise _undefined(request)

        assert (refused.value.code, refused.value.status, refused.value.retryable) == (
            "default_facet_undefined",
            422,
            False,
        )
        assert refused.value.request is request
        assert str(refused.value).endswith(
            f"No UN data was read for {_NGA}, which the UN System Data Commons front door refuses"
        )

    def test_a_non_un_member_binding_no_place_keeps_its_refusal_detail(self):
        graph = _ScopedWithheldGraph(un_places=[_KEN, _NGA])
        undefined = _undefined(_unbound())

        with (
            pytest.raises(DefaultFacetUndefinedError) as refused,
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            note_un_blocked_places([_NGA])
            raise undefined

        assert refused.value is undefined

    def test_a_member_binding_no_place_under_withheld_provenances_refuses_excluded(self):
        graph = _ScopedWithheldGraph(withheld=["dc/base/UN_SDG"])
        # The mixer holds data for B at Kenya and Nigeria, all of it withheld.
        availability = _scope(("B", _KEN), ("B", _NGA))
        deadline = time.monotonic() + 600

        with (
            pytest.raises(ExcludedByRegistry) as refused,
            origin_reads(graph, availability=availability, deadline=deadline),
        ):
            raise _undefined(_unbound("dcid:B"))

        assert refused.value.variables == ("B",)
        assert refused.value.provenances == ("dc/base/UN_SDG",)
        # The re-read covers every place availability names for the member.
        assert graph.calls == [(("B",), (_KEN, _NGA))]
        assert graph.limits == [(deadline, WITHHELD_READ_MAX_BYTES)]
        assert graph.scope_reads == []

    def test_a_member_whose_scope_falls_back_to_countries_re_reads_every_country_and_earth(self):
        # Availability lists too many places, so the scope is B's countries with admitted data,
        # which is none: all its data is withheld. The re-read covers Earth's countries and
        # Earth, every place that fallback could hold.
        graph = _ScopedWithheldGraph(withheld=["dc/base/UN_SDG"], countries=[_KEN, _NGA])
        capped = SdmxCapExceededError("too many places", cap_kind="values", cap=1, observed=2)

        with (
            pytest.raises(ExcludedByRegistry) as refused,
            origin_reads(graph, availability=_FailingAvailability(capped), deadline=None),
        ):
            raise _undefined(_unbound("dcid:B"))

        assert refused.value.variables == ("B",)
        assert graph.calls == [(("B",), (_KEN, _NGA, "Earth"))]

    def test_a_memoised_countries_scope_still_re_reads_every_country_and_earth(self):
        # The first refusal settles B's countries scope; the second takes it from the memo,
        # which carries no countries read, and re-reads Earth's countries all the same.
        graph = _ScopedWithheldGraph(
            withheld=["dc/base/UN_SDG"], countries=[_KEN, _NGA], held=[("B", _KEN)]
        )
        capped = SdmxCapExceededError("too many places", cap_kind="values", cap=1, observed=2)

        for _attempt in range(2):
            with (
                pytest.raises(ExcludedByRegistry),
                origin_reads(graph, availability=_FailingAvailability(capped), deadline=None),
            ):
                raise _undefined(_unbound("dcid:B"))

        assert graph.calls == [(("B",), (_KEN, _NGA, "Earth"))] * 2

    def test_a_member_binding_no_place_with_nothing_withheld_keeps_the_refusal(self):
        graph = _ScopedWithheldGraph(un_places=[_KEN, _NGA])
        availability = _scope(("Count_Person", _KEN))

        with (
            pytest.raises(DefaultFacetUndefinedError),
            origin_reads(graph, availability=availability, deadline=None),
        ):
            raise _undefined(_unbound())

        assert graph.calls == [(("Count_Person",), (_KEN,))]

    def test_a_member_binding_no_place_refuses_upstream_error_when_its_scope_read_fails(self):
        failure = SdmxTransportError("availability read failed")
        graph = _ScopedWithheldGraph(withheld=["dc/base/X"])

        with (
            pytest.raises(PlaceScopeReadError) as refused,
            origin_reads(graph, availability=_FailingAvailability(failure), deadline=None),
        ):
            raise _undefined(_unbound())

        assert refused.value.__cause__ is failure
        assert isinstance(refused.value.__context__, DefaultFacetUndefinedError)
        problem = classify_mint_exception(refused.value)
        assert problem is not None
        assert (problem.status, problem.code, problem.retryable) == (502, "upstream_error", True)
        assert graph.calls == []

    def test_a_un_member_binding_no_place_whose_reads_failed_needs_no_scope_read(self):
        graph = _ScopedWithheldGraph(un_fails=SdmxTransportError("UN availability read failed"))
        request = _unbound(f"dcid:{_UN_VARIABLE}")

        with (
            pytest.raises(UnSystemUnavailable) as refused,
            origin_reads(graph, availability=_scope(), deadline=None),
        ):
            record = un_unavailability_record()
            assert record is not None
            record.add_failure(
                UnReadFailure(variables=(_UN_VARIABLE,), places=(_KEN, "Earth"), detail="503")
            )
            raise _undefined(request)

        assert refused.value.places == (_KEN, "Earth")
        assert graph.scope_reads == []

    def test_a_non_un_member_is_not_blamed_on_refused_places(self):
        with (
            pytest.raises(DefaultFacetUndefinedError),
            origin_reads(_WithheldGraph(), availability=_scope(), deadline=None),
        ):
            note_un_blocked_places([_KEN])
            raise _undefined(_request())

    def test_places_the_un_front_door_refused_become_one_warning(self):
        with origin_reads(_WithheldGraph(), availability=_scope(), deadline=None) as blocked:
            note_un_blocked_places([_KEN, _NGA])
            note_un_blocked_places([_KEN])

        (warning,) = un_unavailable_warnings(blocked)
        assert warning.code == "un_system_unavailable"
        assert warning.message.endswith(f"No UN data was read for {_KEN}, {_NGA}.")

    def test_no_refused_place_warns_nothing(self):
        with origin_reads(_WithheldGraph(), availability=_scope(), deadline=None) as blocked:
            pass

        assert un_unavailable_warnings(blocked) == ()

    def test_a_member_warning_names_only_its_own_refused_places(self):
        with origin_reads(_WithheldGraph(), availability=_scope(), deadline=None) as blocked:
            note_un_blocked_places([_KEN, _UGA])

        un_member = _request(f"dcid:{_UN_VARIABLE}", entities=(_KEN, _NGA))
        (warning,) = un_unavailable_warnings(blocked, request=un_member)
        assert warning.message.endswith(f"No UN data was read for {_KEN}.")
        assert un_unavailable_warnings(blocked, request=_request()) == ()
        # A member binding no place covers every country, so every refused place is its own.
        (unbound_warning,) = un_unavailable_warnings(
            blocked, request=_unbound(f"dcid:{_UN_VARIABLE}")
        )
        assert unbound_warning.message.endswith(f"No UN data was read for {_KEN}, {_UGA}.")


def test_the_mint_door_serves_the_un_refusal_as_a_retryable_upstream_error():
    spec = classify_mint_exception(UnSystemUnavailable(variable=_UN_VARIABLE, places=(_KEN,)))

    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)
    assert spec.detail is not None and _KEN in spec.detail


def test_the_mint_and_definition_doors_serve_the_refusal_as_422():
    refusal = ExcludedByRegistry(variables=("Count_Person",), provenances=("dc/base/UN_SDG",))

    for spec in (
        classify_mint_exception(refusal),
        classify_definition_exception(DefinitionMaterialiseRefused(wrapped=refusal)),
    ):
        assert spec is not None
        assert (spec.status, spec.code, spec.retryable) == (422, "excluded_by_registry", False)
        assert spec.detail is not None and "dc/base/UN_SDG" in spec.detail


def test_the_live_origin_client_implements_the_re_read():
    # `origin_reads` and `withheld_warning` read nothing from a graph without the re-read, so a
    # live client missing it would drop every `excluded_by_registry` and `withheld_by_registry`.
    client = build_graph_client(
        base_url="https://ours.example.org/core/api",
        token=None,
    )
    try:
        assert isinstance(client, WithheldReader)
    finally:
        client.close()
