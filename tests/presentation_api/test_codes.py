"""Checks retry-mode consistency with the baked catalogue and door classifiers."""

from __future__ import annotations

from pathlib import Path

from assembler.catalogue import BakedReleaseRegistry
from assembler.fakes import FakeReleaseRegistry
from presentation_api.codes import (
    CODE_RETRY_MODE,
    SHARE_CODE_CATALOGUE,
    WRITE_SERVED_PLATFORM_CODES,
    retryable_for,
)


def test_retry_mode_mirrors_the_baked_catalogue_manifest(catalogue_dir: Path):
    release = BakedReleaseRegistry(catalogue_dir=catalogue_dir).pinned()
    assert release.retry_modes == CODE_RETRY_MODE


def test_every_single_mode_code_a_door_can_classify_matches_retryable_for(
    release_registry: FakeReleaseRegistry,
):
    """Single-mode codes derive retryability from `retryable_for`.

    Multi-mode codes and `request_invalid` compute or set retryability elsewhere.
    """
    from presentation_api.app import presentation_surface

    codes_computed_at_the_raise_site = {"schema_invalid", "frame_unavailable", "request_invalid"}
    # The share and allowance codes sit outside the component catalogue's manifest; their doors
    # state their retryability themselves, and `test_shares_door` asserts it.
    share_codes = set(SHARE_CODE_CATALOGUE) | WRITE_SERVED_PLATFORM_CODES
    surface = presentation_surface(release_registry=release_registry)
    checked = 0
    for exemplar in surface.exception_exemplars:
        spec = surface.classify(exemplar)
        if spec is None or spec.code in codes_computed_at_the_raise_site | share_codes:
            continue
        assert spec.retryable == retryable_for(spec.code), spec.code
        checked += 1
    assert checked > 0


def test_every_request_invalid_exemplar_reports_retryable_false(
    release_registry: FakeReleaseRegistry,
):
    """`request_invalid` stays non-retryable despite its manifest mode."""
    from presentation_api.app import presentation_surface

    surface = presentation_surface(release_registry=release_registry)
    checked = 0
    for exemplar in surface.exception_exemplars:
        spec = surface.classify(exemplar)
        if spec is None or spec.code != "request_invalid":
            continue
        assert spec.retryable is False
        checked += 1
    assert checked > 0
