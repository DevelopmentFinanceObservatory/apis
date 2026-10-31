"""Recorded client exchanges fit the composed app's schemas and code registry.

`dfo_client_contract` ships the scenarios the DFO Python and TypeScript clients replay, the
response members they act on (`strict-fields.json`) and the caller's next step for each refusal
code (`actions.json`). Each check reads the app's own OpenAPI document, built in process with the
deployed surfaces (Search, Datasets, Graph and Presentation) as the contract's capture tool builds
it, and the code registry the app serves:

- every recorded scenario's API exchange matches a route the app serves, its request body
  validates against that route's request schema, and its response body against the route's
  schema for that status, or the route's default response when it lists none for the status;
- every problem code a recorded response carries is active, and its status is the one the server
  would serve (`dfo_platform.codes.registered_status`): the one status the registry gives the code
  on the route's registry endpoint, or across every endpoint when the route's surface maps the
  route to none in `Surface.registry_endpoints`;
- every strict required member stays required and admits no null, and every strict enum value
  stays admitted;
- every `actions.json` code is an active registry code.

A synthetic scenario edits a recorded one to test a client's handling of a malformed answer, and
a `url` exchange fetches from storage, so neither is checked here.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
from dfo_client_contract import actions, scenarios, strict_fields
from dfo_spec_kit.registry import REGISTRY
from fastapi.routing import APIRoute
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from starlette.routing import Match

import dfo_apis.app as dfo_apis_app
from dfo_platform.app import Surface

# The deployed composition's surface flags, as the capture tool sets them.
_DEPLOYED_FLAGS = {
    "DFO_DATASETS_SURFACE": "1",
    "DFO_GRAPH_SURFACE": "1",
    "DFO_PRESENTATION_SURFACE": "1",
}
_UNDEPLOYED_FLAGS = ("DFO_SKILLS_SURFACE", "DFO_WORKER_SURFACE")
_DOCUMENT_URI = "urn:dfo-apis:openapi"
_PROBLEM_MEDIA_TYPE = "application/problem+json"
# A validation message quotes the failing instance, which can be a whole answer.
_MESSAGE_LIMIT = 300
# Starlette keeps converters on routes, while OpenAPI strips `{id:dfo_spec_id}` to `{id}`.
_CONVERTER_SUFFIX_RE = re.compile(r"\{([^:}]+):[^}]+\}")

_RECORDED = [s for s in scenarios() if s["origin"]["kind"] == "recorded"]
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@dataclass(frozen=True)
class _Served:
    """The composed app's OpenAPI document and the surfaces it was built from."""

    document: dict[str, Any]
    surfaces: tuple[Surface, ...]
    schemas: Registry


@dataclass(frozen=True)
class _Route:
    """The route that serves one recorded request."""

    path: str
    method: str
    endpoint: str | None
    operation: Mapping[str, Any]


@pytest.fixture(scope="session")
def served() -> _Served:
    """Build the deployed composition once and read its document.

    `create_app()` composes `deployed_surfaces()`. The fixture keeps the surfaces so a request's
    route and registry endpoint come from the router that serves it.
    """
    with pytest.MonkeyPatch.context() as patch:
        for name, value in _DEPLOYED_FLAGS.items():
            patch.setenv(name, value)
        for name in _UNDEPLOYED_FLAGS:
            patch.delenv(name, raising=False)
        surfaces = tuple(dfo_apis_app.deployed_surfaces())
        patch.setattr(dfo_apis_app, "deployed_surfaces", lambda: list(surfaces))
        document = dfo_apis_app.create_app().openapi()
    schemas = Registry().with_resource(
        _DOCUMENT_URI, Resource.from_contents(document, default_specification=DRAFT202012)
    )
    return _Served(document=document, surfaces=surfaces, schemas=schemas)


def _pointer(*tokens: str) -> str:
    return "".join("/" + token.replace("~", "~0").replace("/", "~1") for token in tokens)


def _resolve(document: Mapping[str, Any], pointer: str) -> Any:
    node: Any = document
    for token in pointer.removeprefix("/").split("/"):
        node = node[token.replace("~1", "/").replace("~0", "~")]
    return node


def _validator(served: _Served, pointer: str) -> Validator:
    """A validator for the schema at `pointer` in the app's document."""
    return Draft202012Validator({"$ref": f"{_DOCUMENT_URI}#{pointer}"}, registry=served.schemas)


def _causes(error: ValidationError) -> Iterator[ValidationError]:
    """The most specific cause of `error` in each `oneOf` or `anyOf` branch it tried."""
    if not error.context:
        yield error
        return
    branches: dict[str | int, list[ValidationError]] = {}
    for cause in error.context:
        branches.setdefault(cause.relative_schema_path[0], []).append(cause)
    for causes in branches.values():
        if (cause := best_match(causes)) is not None:
            yield from _causes(cause)


def _errors(served: _Served, pointer: str, instance: object) -> list[str]:
    return [
        f"{cause.json_path}: {cause.message[:_MESSAGE_LIMIT]}"
        for error in _validator(served, pointer).iter_errors(instance)
        for cause in _causes(error)
    ]


def _request_path(base_url: str, request: Mapping[str, Any]) -> str | None:
    """The request's path on the app, or `None` for a storage fetch.

    A `path` resolves under the base URL's path and an `origin_path` against its origin."""
    if "path" in request:
        raw = urlsplit(base_url).path.rstrip("/") + request["path"]
    elif "origin_path" in request:
        raw = request["origin_path"]
    else:
        return None
    return unquote(urlsplit(raw).path)


def _route(served: _Served, method: str, path: str) -> _Route | None:
    """The route that serves `method path`, as the app's router picks it."""
    scope = {"type": "http", "method": method, "path": path, "root_path": ""}
    for surface in served.surfaces:
        for route in surface.router.routes:
            if not isinstance(route, APIRoute) or route.matches(scope)[0] is not Match.FULL:
                continue
            schema_path = _CONVERTER_SUFFIX_RE.sub(r"{\1}", route.path)
            operation = served.document["paths"].get(schema_path, {}).get(method.lower())
            if not route.include_in_schema or operation is None:
                return None
            return _Route(
                path=schema_path,
                method=method.lower(),
                endpoint=surface.registry_endpoints.get((method, route.path)),
                operation=operation,
            )
    return None


def _api_exchanges(
    served: _Served, scenario: Mapping[str, Any]
) -> Iterator[tuple[int, Mapping[str, Any], Mapping[str, Any], _Route | None]]:
    """Each API exchange with its index, request, response and the route that serves it."""
    base_url = scenario["origin"]["base_url"]
    for index, exchange in enumerate(scenario["exchanges"]):
        request = exchange["request"]
        path = _request_path(base_url, request)
        if path is None:
            continue
        yield index, request, exchange["response"], _route(served, request["method"], path)


def _media_type(response: Mapping[str, Any]) -> str:
    content_type = response.get("headers", {}).get("content-type", "application/json")
    return content_type.split(";", 1)[0].strip()


def _problem_statuses(code: str, endpoint: str | None) -> frozenset[int]:
    """The statuses the registry gives `code`'s problem details on `endpoint`, or on every
    endpoint when `endpoint` is `None`, as `dfo_platform.codes.registered_status` reads them.

    The server answers with the status only when this set holds exactly one."""
    statuses = (
        mapping.http_status
        for mapping in REGISTRY[code].mappings
        if mapping.channel == "problem_details"
        and mapping.http_status is not None
        and (endpoint is None or mapping.endpoint == endpoint)
    )
    return frozenset(statuses)


def _is_active(code: str) -> bool:
    return code in REGISTRY and REGISTRY[code].lifecycle == "active"


def test_the_contract_records_scenarios(served: _Served):
    assert _RECORDED, "the client contract installs no recorded scenario"
    problems = [
        response
        for scenario in _RECORDED
        for _, _, response, _ in _api_exchanges(served, scenario)
        if _media_type(response) == _PROBLEM_MEDIA_TYPE
    ]
    assert problems, "no recorded API response is a problem document, so no code is checked"


@pytest.mark.parametrize("scenario", _RECORDED, ids=[s["name"] for s in _RECORDED])
def test_recorded_exchanges_match_the_served_document(served: _Served, scenario: Mapping[str, Any]):
    """Each API exchange names a served route, and its bodies validate against that route's
    request schema and its response schema for the recorded status."""
    problems: list[str] = []
    checked = 0
    for index, request, response, route in _api_exchanges(served, scenario):
        checked += 1
        where = f"exchange {index} {request['method']}"
        if route is None:
            problems.append(f"{where}: no route in the served document serves the request")
            continue
        operation_pointer = ("paths", route.path, route.method)

        if "json" in request:
            content = route.operation.get("requestBody", {}).get("content", {})
            if "application/json" not in content:
                problems.append(f"{where} {route.path}: the route takes no JSON body")
            else:
                pointer = _pointer(
                    *operation_pointer, "requestBody", "content", "application/json", "schema"
                )
                problems.extend(
                    f"{where} {route.path} request {error}"
                    for error in _errors(served, pointer, request["json"])
                )

        responses = route.operation["responses"]
        status = str(response["status"])
        key = status if status in responses else "default"
        if key not in responses:
            problems.append(f"{where} {route.path}: no response for status {status}")
            continue
        if "json" not in response:
            continue
        media_type = _media_type(response)
        if media_type not in responses[key].get("content", {}):
            problems.append(f"{where} {route.path}: status {key} serves no {media_type}")
            continue
        pointer = _pointer(*operation_pointer, "responses", key, "content", media_type, "schema")
        problems.extend(
            f"{where} {route.path} response {status} {error}"
            for error in _errors(served, pointer, response["json"])
        )

    assert checked, "the scenario records no API exchange"
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("scenario", _RECORDED, ids=[s["name"] for s in _RECORDED])
def test_recorded_problem_codes_are_registered_on_their_route(
    served: _Served, scenario: Mapping[str, Any]
):
    """Each problem code a response carries is active and answers with the one status the
    registry gives it on the route's endpoint, or across every endpoint when the route maps to
    no endpoint."""
    problems: list[str] = []
    for index, _request, response, route in _api_exchanges(served, scenario):
        if _media_type(response) != _PROBLEM_MEDIA_TYPE or "json" not in response:
            continue
        code = response["json"].get("code")
        path = route.path if route is not None else None
        endpoint = route.endpoint if route is not None else None
        where = f"exchange {index} {path}"
        if code is None or not _is_active(code):
            problems.append(f"{where}: {code!r} is not an active registry code")
            continue
        statuses = sorted(_problem_statuses(code, endpoint))
        scope = f"on {endpoint!r}" if endpoint is not None else "across every endpoint"
        if len(statuses) != 1:
            problems.append(
                f"{where}: the registry gives {code!r} the statuses {statuses} {scope}, so the "
                "server cannot pick one; declare the route's endpoint in its surface's "
                "`registry_endpoints`"
            )
        elif response["status"] != statuses[0]:
            problems.append(
                f"{where}: the registry gives {code!r} status {statuses[0]} {scope}, "
                f"not {response['status']}"
            )
    assert not problems, "\n".join(problems)


_REQUIRED = [entry["pointer"] for entry in strict_fields()["required"]]
_ENUMS = [(entry["pointer"], entry["values"]) for entry in strict_fields()["enums"]]


@pytest.mark.parametrize("pointer", _REQUIRED)
def test_strict_member_stays_required_and_non_nullable(served: _Served, pointer: str):
    parent, separator, member = pointer.rpartition("/properties/")
    assert separator, f"{pointer} names no property"
    assert member in _resolve(served.document, parent).get("required", [])
    assert not _validator(served, pointer).is_valid(None), f"{pointer} admits null"


@pytest.mark.parametrize(("pointer", "values"), _ENUMS, ids=[p for p, _ in _ENUMS])
def test_strict_enum_keeps_every_value(served: _Served, pointer: str, values: list[str]):
    validator = _validator(served, pointer)
    assert [value for value in values if not validator.is_valid(value)] == []


def test_every_action_names_an_active_registry_code():
    codes = [row["code"] for row in actions()["codes"]]
    assert codes
    assert [code for code in codes if not _is_active(code)] == []


def test_the_contract_package_pins_spec_kits_commit():
    """uv forces dfo-data-tools's spec-kit pins onto `dfo-spec-kit`'s commit and nothing forces
    the contract package's, so this test does."""
    sources = tomllib.loads(_PYPROJECT.read_text())["tool"]["uv"]["sources"]
    assert sources["dfo-spec-kit-client-contract"]["rev"] == sources["dfo-spec-kit"]["rev"]
