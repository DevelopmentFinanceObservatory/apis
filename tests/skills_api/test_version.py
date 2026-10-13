"""Perturbs each capability-version input to prove it changes the token."""

from __future__ import annotations

import os
import subprocess
import sys

from skills_api import version
from skills_api.codes import SKILLS_CODE_CATALOGUE, SKILLS_WARNING_CATALOGUE


def _baseline_wire_schemas() -> dict[str, dict]:
    """The default `wire_schemas` set `_compute_skills_capability_version` builds for itself
    when the keyword is omitted, so a perturbation test starts from the real baseline and
    mutates exactly one entry.
    """
    return version.served_wire_schemas()


class TestPerturbationsMoveTheToken:
    def test_code_catalogue_title_change_moves_the_token(self):
        baseline = version._compute_skills_capability_version()
        code, (title, detail) = next(iter(sorted(SKILLS_CODE_CATALOGUE.items())))
        mutated_catalogue = dict(SKILLS_CODE_CATALOGUE)
        mutated_catalogue[code] = (title + " (mutated)", detail)

        perturbed = version._compute_skills_capability_version(code_catalogue=mutated_catalogue)

        assert perturbed != baseline

    def test_code_catalogue_detail_change_moves_the_token(self):
        # The exemplar renderer (`_render_body_exemplars`) supplies an
        # explicit `detail="exemplar detail"` for the first sorted code
        # only, which overrides the catalogue's own detail for that one
        # entry -- so this perturbs a later code, whose rendered detail
        # does come from the catalogue.
        baseline = version._compute_skills_capability_version()
        code, (title, detail) = sorted(SKILLS_CODE_CATALOGUE.items())[-1]
        mutated_catalogue = dict(SKILLS_CODE_CATALOGUE)
        mutated_catalogue[code] = (title, detail + " (mutated)")

        perturbed = version._compute_skills_capability_version(code_catalogue=mutated_catalogue)

        assert perturbed != baseline

    def test_warning_catalogue_title_change_moves_the_token(self):
        baseline = version._compute_skills_capability_version()
        code, (title, detail, severity) = next(iter(sorted(SKILLS_WARNING_CATALOGUE.items())))
        mutated_catalogue = dict(SKILLS_WARNING_CATALOGUE)
        mutated_catalogue[code] = (title + " (mutated)", detail, severity)

        perturbed = version._compute_skills_capability_version(warning_catalogue=mutated_catalogue)

        assert perturbed != baseline

    def test_warning_catalogue_severity_change_moves_the_token(self):
        baseline = version._compute_skills_capability_version()
        code, (title, detail, severity) = next(iter(sorted(SKILLS_WARNING_CATALOGUE.items())))
        other_severity = "degraded" if severity != "degraded" else "warn"
        mutated_catalogue = dict(SKILLS_WARNING_CATALOGUE)
        mutated_catalogue[code] = (title, detail, other_severity)

        perturbed = version._compute_skills_capability_version(warning_catalogue=mutated_catalogue)

        assert perturbed != baseline

    def test_wire_schema_property_change_moves_the_token(self):
        baseline = version._compute_skills_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        mutated_schemas["health_response"] = {
            **mutated_schemas["health_response"],
            "title": mutated_schemas["health_response"].get("title", ""),
            "required": [*mutated_schemas["health_response"].get("required", []), "mutated"],
        }

        perturbed = version._compute_skills_capability_version(wire_schemas=mutated_schemas)

        assert perturbed != baseline

    def test_wire_schema_addition_moves_the_token(self):
        baseline = version._compute_skills_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        mutated_schemas["extra_schema"] = {"type": "object"}

        perturbed = version._compute_skills_capability_version(wire_schemas=mutated_schemas)

        assert perturbed != baseline

    def test_skill_resource_content_schema_change_moves_the_token(self):
        """The resource door's own response model -- proves it is actually
        hashed, not merely present in `_baseline_wire_schemas` by
        coincidence.
        """
        baseline = version._compute_skills_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        mutated_schemas["skill_resource_content"] = {
            **mutated_schemas["skill_resource_content"],
            "required": [
                *mutated_schemas["skill_resource_content"].get("required", []),
                "mutated",
            ],
        }

        perturbed = version._compute_skills_capability_version(wire_schemas=mutated_schemas)

        assert perturbed != baseline

    def test_retyping_the_skill_description_field_moves_the_token(self):
        """`SkillBody.description` is a served field, so its type is hashed even though
        `description` is also an annotation keyword."""
        baseline = version._compute_skills_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        mutated_schemas["skill_body"]["properties"]["description"] = {"type": "integer"}

        perturbed = version._compute_skills_capability_version(wire_schemas=mutated_schemas)

        assert perturbed != baseline


class TestNonPerturbationsDoNotMoveTheToken:
    def test_wire_schema_description_edit_does_not_move_the_token(self):
        baseline = version._compute_skills_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        # Mutate a description -- stripped before hashing
        # (strip_schema_annotations), so this must be a no-op on the token.
        mutated_schemas["health_response"] = {
            **mutated_schemas["health_response"],
            "description": "a reworded, purely cosmetic description",
        }

        unchanged = version._compute_skills_capability_version(wire_schemas=mutated_schemas)

        assert unchanged == baseline


def test_skills_capability_version_is_skills_prefixed_and_deterministic():
    assert version.SKILLS_CAPABILITY_VERSION.startswith("skills_")
    assert version._compute_skills_capability_version() == version.SKILLS_CAPABILITY_VERSION


def test_skills_capability_version_is_stable_across_process_runs():
    """Regression guard: the token must not depend on anything that varies
    per process (an unpinned `trace_id` minted fresh by `get_trace_id`,
    Python's per-process string hash randomization, or similar) -- computed
    in two fresh subprocesses with different hash seeds, so a reintroduced
    unpinned-per-render value would make this flap rather than pass.
    """
    code = "from skills_api.version import SKILLS_CAPABILITY_VERSION as v\nprint(v)\n"
    first = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )
    second = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "2"},
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.strip() == second.stdout.strip()
