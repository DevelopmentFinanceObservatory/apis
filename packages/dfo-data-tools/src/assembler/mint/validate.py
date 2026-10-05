"""Validates minted components and envelopes against catalogue JSON Schemas."""

from __future__ import annotations

from collections.abc import Mapping

import jsonschema

from assembler.catalogue import CatalogueRelease
from assembler.errors import SchemaInvalid, pointer_from_path


def component_registry(release: CatalogueRelease):
    """Resolves the envelope's component `$ref`s to member schemas by `$id`."""
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    resources = [
        Resource.from_contents(schema, default_specification=DRAFT202012)
        for schema in release.component_schemas.values()
    ]
    return Registry().with_resources(
        (resource.id(), resource) for resource in resources if resource.id() is not None
    )


def _validate(
    *,
    schema: Mapping[str, object],
    instance: Mapping[str, object],
    legal_fields: tuple[str, ...],
    registry: object | None = None,
) -> None:
    """Validates minted output. Failures carry schema field names because the payload
    has no submitted-intent field for a precise pointer."""
    validator = (
        jsonschema.Draft202012Validator(schema, registry=registry)
        if registry is not None
        else jsonschema.Draft202012Validator(schema)
    )
    errors = sorted(validator.iter_errors(instance), key=jsonschema.exceptions.relevance)
    if not errors:
        return
    error = errors[0]
    raise SchemaInvalid(
        pointer=pointer_from_path(error.absolute_path),
        detail=f"resolved payload failed structural validation: {error.message}",
        legal_fields=legal_fields,
    )
