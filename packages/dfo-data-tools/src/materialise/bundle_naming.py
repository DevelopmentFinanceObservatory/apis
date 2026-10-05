"""Names, slugs and headings for download bundles.

A bundle is named from its sources (a Datasets bundle) or its card title (a card export), the
date it was first prepared, and a short version tag:

- zip: `dfo_<slug>_<YYYY-MM-DD>_<format>_<tag>.zip`
- bare file: `dfo_<slug>_<YYYY-MM-DD>_<tag>.<ext>`

The tag is `version_tag(content_version)` for a Datasets bundle and `revision_tag(revision)`
for a card export. Slugs are ASCII, lowercase and hyphenated, cut at a word boundary.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from datetime import date

SLUG_MAX_LENGTH = 40
"""The longest slug `slugify` and `sources_slug` return."""

RESERVED_MEMBER_SLUGS = frozenset({"codes"})
"""Slugs the bundle's codes table holds, which no collection member takes."""

EN_DASH = "\u2013"
"""The dash between the first and last year of a window."""

_FALLBACK_SLUG = "dataset"
_CARD_FALLBACK_SLUG = "card"
_PARENTHETICAL = re.compile(r"\s*\([^()]*\)")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_VERSION_PREFIX = "cv_"
_VERSION_TAG_LENGTH = 6


def slugify(text: str, *, max_length: int = SLUG_MAX_LENGTH) -> str:
    """`text` as ASCII lowercase words joined by hyphens, at most `max_length` characters.

    Accents fold to their base letter and other non-ASCII characters drop. The cut falls at
    the last hyphen that fits. A first word longer than `max_length` is cut inside the word.
    Returns an empty string when `text` holds no ASCII letter or digit."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _NON_ALPHANUMERIC.sub("-", ascii_text.lower()).strip("-")
    if len(slug) <= max_length:
        return slug
    cut = slug[: max_length + 1]
    boundary = cut.rfind("-")
    return (cut[:boundary] if boundary > 0 else slug[:max_length]).strip("-")


def sources_slug(source_names: Sequence[str]) -> str:
    """The slug naming a bundle by its sources, at most `SLUG_MAX_LENGTH` characters.

    Parentheticals drop ("OECD Creditor Reporting System (CRS)" gives
    `oecd-creditor-reporting-system`). One source gives its own slug, two give
    `<a>-and-<b>`, and more give `<a>-and-<n>-more`, where `n` counts the other sources.
    Repeated names count once. Returns `dataset` when no name yields a slug."""
    names = [slug for name in dict.fromkeys(source_names) if (slug := _name_slug(name))]
    if not names:
        return _FALLBACK_SLUG
    first = names[0]
    if len(names) == 1:
        return slugify(first)
    if len(names) > 2:
        suffix = f"-and-{len(names) - 1}-more"
        return slugify(first, max_length=SLUG_MAX_LENGTH - len(suffix)) + suffix
    # The first name gets at least half the budget, the second what that leaves, and the
    # first then takes back what the second did not use.
    second = names[1]
    budget = SLUG_MAX_LENGTH - len("-and-")
    cut_first = slugify(first, max_length=max(budget - len(second), budget // 2))
    second = slugify(second, max_length=budget - len(cut_first))
    first = slugify(first, max_length=budget - len(second))
    return f"{first}-and-{second}"


def card_slug(title: str) -> str:
    """The slug naming a card export by its card title, or `card` when the title yields none."""
    return slugify(title) or _CARD_FALLBACK_SLUG


def _name_slug(name: str) -> str:
    return slugify(_PARENTHETICAL.sub("", name), max_length=len(name))


def heading(*, source_names: Sequence[str], start_year: int | None, end_year: int | None) -> str:
    """The heading of a Datasets bundle, from its sources and the years its data covers.

    One source gives `<A>, <start>-<end>`, two give `<A> and <B>, …`, and more give
    `<A> and <n> more sources, …`. Repeated names count once. A window of one year prints that
    year alone, and a missing year on either end omits the years. The years join with an en
    dash. Source names keep their parentheticals."""
    names = list(dict.fromkeys(source_names))
    if not names:
        raise ValueError("a bundle heading needs at least one source name")
    if len(names) == 1:
        title = names[0]
    elif len(names) == 2:
        title = f"{names[0]} and {names[1]}"
    else:
        title = f"{names[0]} and {len(names) - 1} more sources"
    if start_year is None or end_year is None:
        return title
    return f"{title}, {years_text(start_year, end_year)}"


def years_text(start: int, end: int) -> str:
    """A year window as text: `2020` for one year, `1973\u20132024` for several."""
    return str(start) if start == end else f"{start}{EN_DASH}{end}"


def version_tag(content_version: str) -> str:
    """The first six characters of a content version after its `cv_` prefix:
    `cv_5c9dfd37…` gives `5c9dfd`."""
    return content_version.removeprefix(_VERSION_PREFIX)[:_VERSION_TAG_LENGTH]


def revision_tag(revision: int) -> str:
    """A card revision's tag: revision 3 gives `r3`."""
    return f"r{revision}"


def zip_file_name(*, slug: str, prepared: date, format: str, tag: str) -> str:
    """`dfo_<slug>_<YYYY-MM-DD>_<format>_<tag>.zip`, the download name of a bundle."""
    return f"dfo_{slug}_{prepared.isoformat()}_{format}_{tag}.zip"


def bare_file_name(
    *, slug: str, prepared: date, tag: str, extension: str, kind: str | None = None
) -> str:
    """`dfo_<slug>_<YYYY-MM-DD>_<tag>.<extension>`, the download name of a file served alone.

    A `kind` names a file that is not the data itself, in the format's place:
    `dfo_<slug>_<YYYY-MM-DD>_<kind>_<tag>.<extension>`."""
    kind_part = f"_{kind}" if kind is not None else ""
    return f"dfo_{slug}_{prepared.isoformat()}{kind_part}_{tag}.{extension}"


def member_slugs(member_source_names: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """One slug per collection member, in order, from each member's source names.

    A slug an earlier member took, or one in `RESERVED_MEMBER_SLUGS`, gains the first free
    suffix `-2`, `-3`, and so on."""
    taken: set[str] = set(RESERVED_MEMBER_SLUGS)
    slugs: list[str] = []
    for names in member_source_names:
        base = sources_slug(names)
        slug = base
        suffix = 2
        while slug in taken:
            slug = f"{base}-{suffix}"
            suffix += 1
        taken.add(slug)
        slugs.append(slug)
    return tuple(slugs)
