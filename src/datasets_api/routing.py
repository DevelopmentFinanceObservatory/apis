"""Registers case-sensitive `spec_`/`col_` path convertors and fallback routes for malformed IDs.
The observations route takes `spec_` ids alone, so a `col_` id there matches no route.
Each fallback excludes literals reserved by its route shape, then returns a grammar-naming 404 for
every HTTP method. Starlette's dispatcher treats this like any other routing 404. Import this module
before building routes. Repeat registration uses the same process-global keys.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.convertors import Convertor, register_url_convertor


class SubjectIdConvertor(Convertor[str]):
    """Matches one `spec_`- or `col_`-prefixed path segment, verbatim."""

    regex = r"(?:spec_|col_)[^/]+"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("dfo_subject_id", SubjectIdConvertor())


class SpecIdConvertor(Convertor[str]):
    """Matches one `spec_`-prefixed path segment, verbatim."""

    regex = r"spec_[^/]+"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("dfo_spec_id", SpecIdConvertor())


class MalformedSubjectIdConvertor(Convertor[str]):
    """Matches segments outside the subject grammar and reserved route literals. Each fallback stays
    disjoint from real routes."""

    def __init__(self, reserved_segments: tuple[str, ...]) -> None:
        exclude_reserved = (
            r"(?!(?:" + "|".join(reserved_segments) + r")(?:/|$))" if reserved_segments else ""
        )
        self.regex: str = exclude_reserved + r"(?!(?:spec_|col_)[^/]+(?:/|$))" r"[^/]+"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor(
    "dfo_malformed_subject_id_excluding_meta_literals",
    MalformedSubjectIdConvertor(("register", "healthz", "schema", "capabilities")),
)
register_url_convertor(
    "dfo_malformed_subject_id_excluding_operations",
    MalformedSubjectIdConvertor(("operations",)),
)
register_url_convertor(
    "dfo_malformed_subject_id",
    MalformedSubjectIdConvertor(()),
)


def _grammar_refusal_detail(segment: str) -> str:
    return f"{segment!r} is not a valid dataset id: an id is `spec_`- or `col_`-prefixed, case-sensitive"


_ALL_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


def build_id_grammar_refusal_router() -> APIRouter:
    """Adds fallback routes for five id-taking shapes, each reserving its colliding literals.
    Every method returns 404 for malformed IDs. Starlette prefers a later full path-and-method match
    over an earlier partial match, so a GET-only fallback could return 405 for POST."""
    router = APIRouter()

    async def refuse_malformed_id(id: str, content_version: str | None = None) -> None:
        raise HTTPException(status_code=404, detail=_grammar_refusal_detail(id))

    for path in (
        "/api/datasets/{id:dfo_malformed_subject_id_excluding_meta_literals}",
        "/api/datasets/{id:dfo_malformed_subject_id_excluding_operations}/options",
        "/api/datasets/{id:dfo_malformed_subject_id_excluding_operations}/{content_version}",
        "/api/datasets/{id:dfo_malformed_subject_id}/{content_version}/pin",
        "/api/datasets/{id:dfo_malformed_subject_id}/{content_version}/observations",
    ):
        router.add_api_route(
            path,
            refuse_malformed_id,
            methods=list(_ALL_METHODS),
            include_in_schema=False,
        )

    return router
