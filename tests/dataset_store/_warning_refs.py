"""Stored warnings in every format migration 0039 reads, for the live tests of 0039 and of
`dataset_store.warning_formats`.

Each old format is the message a writer stored before warnings counted their refs. The new
shapes come from the current writers in `materialise.finish`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from urllib.parse import urlsplit

from dataset_store import warning_formats
from materialise import finish as finish_module
from materialise.codes import MATERIALISE_CODE_CATALOGUE
from materialise.finish import (
    currency_unmapped_omitted_warning,
    deflator_missing_omitted_warning,
)

FALLBACK = (
    " No place was named, so the answer covers the countries with data, because the full "
    "list of places with data was too large or unavailable."
)
ENTITIES = ["country/BDI", "country/KEN", "country/RWA", "country/SSD"]
UNSERVED = [*ENTITIES, "country/TZA", "country/UGA", "country/ZMB"]
MEMBERS = ["dcid:YouthRate", "dcid:AdultRate", "dcid:ElderRate"]
MANY_MEMBERS = [f"dcid:Member{index}" for index in range(6)]
OMITTED = [f"country/C{index:02}" for index in range(12)]
DEFLATED = [("country/KEN", year) for year in [*range(1990, 2002), 2004, 2007, 2008]] + [
    ("country/UGA", 2010)
]


def _old_names(names: list[str]) -> str:
    """The first five of `names` with a count of the rest, as warnings named them before they
    counted (`materialise.finish._capped_names`)."""
    rest = len(names) - 5
    return ", ".join(names[:5]) + (f" and {rest} more" if rest > 0 else "")


def _old_coverage(
    *,
    unserved: Sequence[str] = (),
    members: Sequence[str] = (),
    range_window: bool = False,
) -> str:
    """A `coverage_below_scope` message as the writer stored it before it counted its refs."""
    _title, message = MATERIALISE_CODE_CATALOGUE["coverage_below_scope"]
    if unserved:
        message += f" No observations were served for {_old_names(list(unserved))}."
    if members:
        message += (
            f" No observations were served for {_old_names(list(members))}, which hold none at "
            "the observation period the other members serve"
            f"{' in the requested years' if range_window else ''}."
        )
    return message


def _old_omitted(code: str, named: list[str], *, count: int) -> dict[str, object]:
    """A conversion warning as the writer stored it before it counted its entities: `count`
    entities, the first ten `named`."""
    _title, detail = MATERIALISE_CODE_CATALOGUE[code]
    truncated = " (truncated)" if count > 10 else ""
    return {
        "code": code,
        "severity": "warn",
        "message": f"{detail} {count} entities: {', '.join(named[:10])}{truncated}",
    }


def new_coverage(
    *,
    unserved: Sequence[str] = (),
    members: Sequence[str] = (),
    member_count: int = 0,
    countries_fallback: bool = False,
) -> dict[str, object]:
    return dict(
        finish_module._coverage_below_scope_warning(
            unserved,
            countries_fallback=countries_fallback,
            off_cadence=members,
            out_of_window=(),
            member_count=member_count,
        )
    )


def warning_refs_body(warnings: list[dict[str, object]], *, members: int | None) -> dict:
    """A version body holding `warnings`, its trace resolving `members` StatVars, or holding no
    `resolved_stat_vars` with `None`."""
    trace: dict[str, object] = {"graph_revision": "rev"}
    if members is not None:
        trace["resolved_stat_vars"] = [
            {"ref": f"dcid:Member{index}", "roles": ["recipient"], "observation_count": 0}
            for index in range(members)
        ]
    return {"prepared_at": "2026-10-20T00:00:00Z", "trace": trace, "warnings": warnings}


def warning_refs_cases() -> dict[str, tuple[dict, list[dict[str, object]]]]:
    """Version bodies holding every stored warning format 0039 reads, by case, each with the
    warnings 0039 leaves in its row."""
    empty = {"code": "empty_result", "severity": "warn", "message": "No rows."}
    deflator_detail = MATERIALISE_CODE_CATALOGUE["deflator_missing_omitted"][1]
    cases: dict[str, tuple[dict, list[dict[str, object]]]] = {
        "unserved_with_fallback": (
            warning_refs_body(
                [
                    {
                        "code": "coverage_below_scope",
                        "severity": "warn",
                        "message": _old_coverage(unserved=UNSERVED) + FALLBACK,
                    }
                ],
                members=None,
            ),
            [new_coverage(unserved=UNSERVED, countries_fallback=True)],
        ),
        "unserved_and_members": (
            warning_refs_body(
                [
                    {
                        "code": "coverage_below_scope",
                        "severity": "warn",
                        "message": _old_coverage(unserved=UNSERVED[:1], members=MEMBERS),
                    },
                    empty,
                ],
                members=7,
            ),
            [
                new_coverage(
                    unserved=UNSERVED[:1],
                    members=MEMBERS,
                    member_count=7,
                ),
                empty,
            ],
        ),
        "range_members_with_fallback": (
            warning_refs_body(
                [
                    {
                        "code": "coverage_below_scope",
                        "severity": "warn",
                        "message": _old_coverage(members=MANY_MEMBERS, range_window=True)
                        + FALLBACK,
                    }
                ],
                members=9,
            ),
            [new_coverage(members=MANY_MEMBERS, member_count=9, countries_fallback=True)],
        ),
        "members_without_trace_count": (
            warning_refs_body(
                [
                    {
                        "code": "coverage_below_scope",
                        "severity": "warn",
                        "message": _old_coverage(members=MEMBERS),
                    }
                ],
                members=None,
            ),
            [
                {
                    "code": "coverage_below_scope",
                    "severity": "warn",
                    "message": MATERIALISE_CODE_CATALOGUE["coverage_below_scope"][1]
                    + " 3 series report at a different frequency and are left out of the "
                    "figures.",
                    "off_cadence_members": {"count": 3, "refs": MEMBERS},
                }
            ],
        ),
        "currency_truncated": (
            warning_refs_body(
                [
                    _old_omitted(
                        "currency_unmapped_omitted",
                        OMITTED,
                        count=len(OMITTED),
                    )
                ],
                members=None,
            ),
            [dict(currency_unmapped_omitted_warning(OMITTED))],
        ),
        "currency_one_entity": (
            warning_refs_body(
                [_old_omitted("currency_unmapped_omitted", ["country/KEN"], count=1)],
                members=None,
            ),
            [dict(currency_unmapped_omitted_warning(["country/KEN"]))],
        ),
        "deflator_spans": (
            warning_refs_body(
                [
                    {
                        "code": "deflator_missing_omitted",
                        "severity": "warn",
                        "message": f"{deflator_detail} 2 entities: "
                        "country/KEN (1990-2001, 2004, 2007-2008), country/UGA (2010)",
                    }
                ],
                members=None,
            ),
            [dict(deflator_missing_omitted_warning(DEFLATED))],
        ),
        "deflator_truncated": (
            warning_refs_body(
                [
                    {
                        "code": "deflator_missing_omitted",
                        "severity": "warn",
                        "message": f"{deflator_detail} 12 entities: "
                        + ", ".join(f"{entity} (2001-2002)" for entity in OMITTED[:10])
                        + " (truncated)",
                    }
                ],
                members=None,
            ),
            [
                dict(
                    deflator_missing_omitted_warning(
                        (entity, year) for entity in OMITTED for year in (2001, 2002)
                    )
                )
            ],
        ),
    }
    # A row the writer stored between the member fields and the entity fields: its member
    # sentence already counts, its unserved sentence still names entities.
    members_counted = new_coverage(members=MEMBERS, member_count=4)
    _title, coverage_detail = MATERIALISE_CODE_CATALOGUE["coverage_below_scope"]
    cases["unserved_beside_counted_members"] = (
        warning_refs_body(
            [
                {
                    **members_counted,
                    "message": cast(str, members_counted["message"]).replace(
                        coverage_detail,
                        _old_coverage(unserved=ENTITIES),
                    ),
                }
            ],
            members=4,
        ),
        [new_coverage(unserved=ENTITIES, members=MEMBERS, member_count=4)],
    )
    current = [
        new_coverage(
            unserved=UNSERVED,
            members=MEMBERS,
            member_count=4,
            countries_fallback=True,
        ),
        new_coverage(unserved=["country/KEN"]),
        dict(currency_unmapped_omitted_warning(OMITTED)),
        dict(deflator_missing_omitted_warning(DEFLATED)),
        empty,
    ]
    cases["current"] = (warning_refs_body(current, members=4), current)
    return cases


def format_count_bodies() -> dict[str, dict]:
    """The bodies of `warning_refs_cases`, and three more: a counted member warning whose
    `member_count` differs from its trace, one whose trace holds no `resolved_stat_vars`, and a
    currency warning holding only its detail."""
    bodies = {name: body for name, (body, _expected) in warning_refs_cases().items()}
    counted = new_coverage(members=MEMBERS, member_count=5)
    _title, currency_detail = MATERIALISE_CODE_CATALOGUE["currency_unmapped_omitted"]
    bodies["member_count_differs"] = warning_refs_body([counted], members=3)
    bodies["member_count_without_trace"] = warning_refs_body([counted], members=None)
    bodies["currency_detail_only"] = warning_refs_body(
        [{"code": "currency_unmapped_omitted", "severity": "warn", "message": currency_detail}],
        members=None,
    )
    return bodies


def format_counts(dsn: str, monkeypatch, capsys) -> dict[str, tuple[int | None, int]]:
    """The counts `python -m dataset_store.warning_formats` prints against the store at `dsn`,
    by format: its warnings, `None` for the rewritten-rows line, and its version rows."""
    parts = urlsplit(dsn)
    for name, value in {
        "DATASET_STORE_HOST": parts.hostname,
        "DATASET_STORE_PORT": str(parts.port or 5432),
        "DATASET_STORE_DATABASE": parts.path.lstrip("/"),
        "DATASET_STORE_MIGRATOR_DB_USER": parts.username,
        "DATASET_STORE_MIGRATOR_DB_PASSWORD": parts.password,
    }.items():
        monkeypatch.setenv(name, cast(str, value))
    monkeypatch.delenv("DATASET_STORE_INSTANCE_CONNECTION_NAME", raising=False)
    capsys.readouterr()
    assert warning_formats.main() == 0
    counts: dict[str, tuple[int | None, int]] = {}
    for line in capsys.readouterr().out.splitlines():
        name, _, value = line.rpartition(": ")
        warnings, _, rows = value.partition(" warnings in ")
        counts[name] = (None, int(warnings)) if not rows else (int(warnings), int(rows.split()[0]))
    return counts


def counts_since(
    before: dict[str, tuple[int | None, int]], after: dict[str, tuple[int | None, int]]
) -> dict[str, tuple[int | None, int]]:
    """Each format's counts in `after` less those in `before`."""
    return {
        name: (
            None if warnings is None else warnings - cast(int, before[name][0]),
            rows - before[name][1],
        )
        for name, (warnings, rows) in after.items()
    }
