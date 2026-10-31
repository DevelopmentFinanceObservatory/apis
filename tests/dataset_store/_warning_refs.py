"""Stored warnings in every format `dataset_store.warning_formats` lists, for the tests of its
dry run and of `warning_rewrite.sql`.

Each old format is the message a writer stored before warnings counted their refs. The new
shapes come from the current writers in `materialise.finish` and `datasets_api.origin`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from datasets_api.origin import _withheld_warning
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from materialise import finish as finish_module
from materialise.codes import MATERIALISE_CODE_CATALOGUE
from materialise.finish import (
    currency_unmapped_omitted_warning,
    deflator_missing_omitted_warning,
    deflator_projected_warning,
)

type StoredWarning = dict[str, object]

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
COVERAGE_DETAIL = MATERIALISE_CODE_CATALOGUE["coverage_below_scope"][1]


@dataclass(frozen=True)
class _Projected:
    projected_count: int
    projected_examples: Sequence[str]


def _old_names(names: Sequence[str]) -> str:
    """The first five of `names` with a count of the rest, as warnings named them before they
    counted (`dfo_platform.named.named_capped`)."""
    rest = len(names) - 5
    return ", ".join(names[:5]) + (f" and {rest} more" if rest > 0 else "")


def old_unserved_sentence(entities: Sequence[str]) -> str:
    return f" No observations were served for {_old_names(entities)}."


def old_member_sentence(members: Sequence[str], *, range_window: bool = False) -> str:
    return (
        f" No observations were served for {_old_names(members)}, which hold none at the "
        "observation period the other members serve"
        f"{' in the requested years' if range_window else ''}."
    )


def _coverage(message: str, **fields: object) -> StoredWarning:
    return {"code": "coverage_below_scope", "severity": "warn", "message": message, **fields}


def _old_omitted(code: str, named: Sequence[str], *, count: int) -> StoredWarning:
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
) -> StoredWarning:
    return dict(
        finish_module._coverage_below_scope_warning(
            unserved,
            countries_fallback=countries_fallback,
            off_cadence=members,
            out_of_window=(),
            unserved_members=(),
            member_count=member_count,
        )
    )


def current_warnings() -> list[StoredWarning]:
    """Warnings in the shapes today's writers produce, which no format matches."""
    return [
        new_coverage(unserved=UNSERVED, members=MEMBERS, member_count=4, countries_fallback=True),
        new_coverage(unserved=["country/KEN"]),
        dict(currency_unmapped_omitted_warning(OMITTED)),
        dict(currency_unmapped_omitted_warning(["country/KEN"])),
        dict(deflator_missing_omitted_warning(DEFLATED)),
        {"code": "empty_result", "severity": "warn", "message": "No rows."},
    ]


def rewrite_cases() -> dict[str, tuple[list[StoredWarning], list[StoredWarning]]]:
    """Stored warning arrays by case, each with the array `warning_rewrite.sql` leaves."""
    deflator_detail = MATERIALISE_CODE_CATALOGUE["deflator_missing_omitted"][1]
    un_detail = DATASETS_WARNING_CATALOGUE["un_system_unavailable"][1]
    unserved_one = cast(str, new_coverage(unserved=UNSERVED[:1])["message"])
    members_counted = new_coverage(members=MEMBERS, member_count=4)
    unchanged: dict[str, list[StoredWarning]] = {
        "range_members_with_fallback": [
            _coverage(
                COVERAGE_DETAIL + old_member_sentence(MANY_MEMBERS, range_window=True) + FALLBACK
            )
        ],
        "name_with_comma": [
            _coverage(COVERAGE_DETAIL + old_unserved_sentence(["country/KEN", "country/A,B"]))
        ],
        "withheld": [
            _withheld_warning("dcid:Rate", {"country/KEN": ("dc/p1",)}).model_dump(
                exclude_none=True
            )
        ],
        "un_unread": [
            {
                "code": "un_system_unavailable",
                "severity": "warn",
                "message": f"{un_detail} No UN data was read for country/KEN, country/UGA.",
            }
        ],
        "projected": [dict(deflator_projected_warning([_Projected(2, ["country/KEN@2020"])]))],
        "unlisted": [
            {
                "code": "facet_default_selected",
                "severity": "warn",
                "message": "selected 'dc/abc' by its ranking",
            },
            {
                "code": "stat_vars_summed",
                "severity": "info",
                "message": "3 StatVars were summed into dcid:Family.",
            },
        ],
        "current": current_warnings(),
    }
    return {
        "unserved_with_fallback": (
            [_coverage(COVERAGE_DETAIL + old_unserved_sentence(UNSERVED) + FALLBACK)],
            [new_coverage(unserved=UNSERVED, countries_fallback=True)],
        ),
        "unserved_and_members": (
            [
                _coverage(
                    COVERAGE_DETAIL
                    + old_unserved_sentence(UNSERVED[:1])
                    + old_member_sentence(MEMBERS)
                ),
                {"code": "empty_result", "severity": "warn", "message": "No rows."},
            ],
            [
                {
                    **new_coverage(unserved=UNSERVED[:1]),
                    "message": unserved_one + old_member_sentence(MEMBERS),
                },
                {"code": "empty_result", "severity": "warn", "message": "No rows."},
            ],
        ),
        # A row stored between the member fields and the entity fields: its member sentence
        # already counts, its unserved sentence still names entities.
        "unserved_beside_counted_members": (
            [
                {
                    **members_counted,
                    "message": cast(str, members_counted["message"]).replace(
                        COVERAGE_DETAIL, COVERAGE_DETAIL + old_unserved_sentence(ENTITIES)
                    ),
                }
            ],
            [new_coverage(unserved=ENTITIES, members=MEMBERS, member_count=4)],
        ),
        "currency_truncated": (
            [_old_omitted("currency_unmapped_omitted", OMITTED, count=len(OMITTED))],
            [dict(currency_unmapped_omitted_warning(OMITTED))],
        ),
        "currency_one_entity": (
            [_old_omitted("currency_unmapped_omitted", ["country/KEN"], count=1)],
            [dict(currency_unmapped_omitted_warning(["country/KEN"]))],
        ),
        "deflator_spans": (
            [
                {
                    "code": "deflator_missing_omitted",
                    "severity": "warn",
                    "message": f"{deflator_detail} 2 entities: "
                    "country/KEN (1990-2001, 2004, 2007-2008), country/UGA (2010)",
                }
            ],
            [dict(deflator_missing_omitted_warning(DEFLATED))],
        ),
        "deflator_truncated": (
            [
                {
                    "code": "deflator_missing_omitted",
                    "severity": "warn",
                    "message": f"{deflator_detail} 12 entities: "
                    + ", ".join(f"{entity} (2001-2002)" for entity in OMITTED[:10])
                    + " (truncated)",
                }
            ],
            [
                dict(
                    deflator_missing_omitted_warning(
                        (entity, year) for entity in OMITTED for year in (2001, 2002)
                    )
                )
            ],
        ),
        **{name: (warnings, warnings) for name, warnings in unchanged.items()},
    }
