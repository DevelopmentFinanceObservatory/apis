"""Map a `licenseType` dcid to the licence identifiers a download bundle writes.

`datapackage.json` `licenses[].name` takes an Open Definition licence id, and
`CITATION.cff` `license` takes an SPDX id. The two lists differ. The Open
Definition list has no `CC-BY-2.5`, `CC-BY-3.0-IGO` or CDLA ids, and spells
`ODC-BY-1.0` where SPDX writes `ODC-By-1.0`. Any other licence is written by
its graph URL and label. `PublicDomain` maps to neither, because `PDDL-1.0` is a
specific dedication its publishers did not make.
"""

from __future__ import annotations

from typing import NamedTuple


class LicenceIds(NamedTuple):
    """The ids one `licenseType` maps to, `None` where the list has none."""

    open_definition: str | None
    spdx: str | None


PUBLIC_DOMAIN = "PublicDomain"
"""The `licenseType` dcid for a public-domain source."""

_IDS: dict[str, LicenceIds] = {
    "CreativeCommons0": LicenceIds("CC0-1.0", "CC0-1.0"),
    "OpenDataCommonsOpenDatabaseLicense": LicenceIds("ODbL-1.0", "ODbL-1.0"),
    "OpenDataCommonsAttributionLicense": LicenceIds("ODC-BY-1.0", "ODC-By-1.0"),
    "CommunityDataLicenseAgreementPermissive1.0": LicenceIds(None, "CDLA-Permissive-1.0"),
    "CommunityDataLicenseAgreementSharing1.0": LicenceIds(None, "CDLA-Sharing-1.0"),
    "CreativeCommonsAttributionIntergovernmentalOrganizationLicense": LicenceIds(
        None, "CC-BY-3.0-IGO"
    ),
    "CreativeCommonsAttributionLicenseGeneric": LicenceIds(None, "CC-BY-2.5"),
}

_NO_IDS = LicenceIds(None, None)


def licence_ids(licence_type: str | None) -> LicenceIds:
    """The Open Definition and SPDX ids of `licence_type`, both `None` when unmapped."""
    if licence_type is None:
        return _NO_IDS
    return _IDS.get(licence_type, _NO_IDS)


def licence_title(licence_type: str | None, label: str | None) -> str | None:
    """The display title of a licence: "Public domain" for `PublicDomain`, else `label`."""
    if licence_type == PUBLIC_DOMAIN:
        return "Public domain"
    return label
