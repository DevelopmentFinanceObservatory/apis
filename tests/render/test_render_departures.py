"""Pins departure text: the exact sentence for each code on a part that shows data and on
one that shows none, and the second person over every code."""

from __future__ import annotations

import re

import pytest

from search.heading.departure_fact import DepartureFact, UnservedMembers
from search.models import DEPARTURE_ORDER, Departure
from search.render.departures import (
    departure_text,
    departures_from_facts,
    no_data_limit_text,
)

_CASES: list[tuple[DepartureFact, str | None, str]] = [
    (
        DepartureFact(
            code="closest_match",
            binding_ref="what:",
            asked="cost of living",
            shown="healthcare expenditure per person",
        ),
        None,
        "You asked for cost of living. This shows healthcare expenditure per person, the "
        "closest measure the data holds.",
    ),
    # The elected measure holds no data, so several readings stand in for it.
    (
        DepartureFact(
            code="closest_match", binding_ref=None, asked="child mortality", readings=True
        ),
        None,
        "You asked for child mortality. These readings are the closest measures the data holds.",
    ),
    # A resubmit holds no words for the measure, so the text states only what is shown.
    (
        DepartureFact(code="closest_match", binding_ref=None, shown="neonatal mortality rate"),
        None,
        "This shows neonatal mortality rate, the closest measure the data holds.",
    ),
    (
        DepartureFact(code="closest_match", binding_ref=None, readings=True),
        None,
        "These readings are the closest measures the data holds.",
    ),
    # A resubmit's posted value the data cannot meet, named without quoting the request.
    (
        DepartureFact(
            code="measure_not_applied",
            binding_ref="how:crsTypeOfFinance",
            values=("debt instruments",),
        ),
        "aid",
        "This shows the measure without debt instruments.",
    ),
    # A dropped phrase is the user's own words, quoted whatever it opens with (mls-030).
    (
        DepartureFact(
            code="measure_not_applied", binding_ref="how:crsFinanceType", asked="grant equivalent"
        ),
        "aid",
        'You asked for "grant equivalent". This shows the measure without it.',
    ),
    (
        DepartureFact(
            code="measure_not_applied",
            binding_ref=None,
            asked="through regional development banks",
        ),
        "aid",
        'You asked for "through regional development banks". This shows the measure without it.',
    ),
    # crs-038: a price basis is cleared by the unit token, which states the prices shown,
    # and is stated without its leading preposition.
    (
        DepartureFact(
            code="measure_not_applied",
            binding_ref="unit:",
            asked="in constant 2022 dollars",
            shown="current US dollars",
        ),
        "aid",
        "You asked for constant 2022 dollars. This shows current US dollars.",
    ),
    (
        DepartureFact(
            code="measure_not_applied",
            binding_ref="unit:",
            asked="constant 2022 dollars",
            shown="current US dollars",
        ),
        "aid",
        "You asked for constant 2022 dollars. This shows current US dollars.",
    ),
    # crs-040: the share is named by its verb; the recorded phrase is a fragment.
    (
        DepartureFact(
            code="operation_not_applied",
            binding_ref=None,
            asked="share of ODA going to Kenya out of",
            verb="share_of",
        ),
        "aid",
        "You asked for a share. This shows the amounts behind it, because Search does not "
        "compute shares.",
    ),
    (
        DepartureFact(
            code="operation_not_applied",
            binding_ref=None,
            asked="Which donors give the most",
            verb="rank",
        ),
        "aid",
        "You asked for a ranking. This shows the amounts behind it, because Search does not "
        "compute rankings.",
    ),
    # ids-006: the role noun comes from the flow family.
    (
        DepartureFact(
            code="breakdown_not_applied", binding_ref="origin:", asked="by creditor", over="origin"
        ),
        "debt_stock",
        "You asked for a breakdown by creditor. This is the total; name a creditor to see one.",
    ),
    (
        DepartureFact(
            code="breakdown_not_applied",
            binding_ref="where:",
            asked="per recipient",
            over="destination",
        ),
        "aid",
        "You asked for a breakdown by recipient. This is the total; name a recipient to see one.",
    ),
    # crs-043: a named year past the data.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="2030",
            shown="1973 to 2024",
            requested_years=(2030, 2030),
            data_years=(1973, 2024),
        ),
        "aid",
        "You asked for 2030. The data ends in 2024, so this shows 1973 to 2024.",
    ),
    # A resubmit's window, stated without quoting it.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            shown="1973 to 2024",
            requested_years=(2030, 2030),
            data_years=(1973, 2024),
        ),
        "aid",
        "The data ends in 2024, so this shows 1973 to 2024.",
    ),
    # A range partly before the data.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="1960 to 2020",
            shown="1973 to 2020",
            requested_years=(1960, 2020),
            data_years=(1973, 2024),
        ),
        "aid",
        "You asked for 1960 to 2020. The data starts in 1973, so this shows 1973 to 2020.",
    ),
    # A range partly after the data.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="2020 to 2030",
            shown="2020 to 2024",
            requested_years=(2020, 2030),
            data_years=(1973, 2024),
        ),
        "aid",
        "You asked for 2020 to 2030. The data ends in 2024, so this shows 2020 to 2024.",
    ),
    # A range past both ends.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="1950 to 2030",
            shown="1973 to 2024",
            requested_years=(1950, 2030),
            data_years=(1973, 2024),
        ),
        "aid",
        "You asked for 1950 to 2030. The data covers only 1973 to 2024.",
    ),
    # A no-data answer (`shown` is None): a year before the data, mls-020.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="2010",
            requested_years=(2010, 2010),
            data_years=(2011, 2024),
        ),
        None,
        "You asked for 2010. The data covers only 2011 to 2024.",
    ),
    # A no-data answer for an open range past the data.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="2035 or later",
            requested_years=(2035, None),
            data_years=(2000, 2004),
        ),
        None,
        "You asked for 2035 or later. The data covers only 2000 to 2004.",
    ),
    # A no-data answer for years inside the span that hold no point.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="2015",
            requested_years=(2015, 2015),
            data_years=(2010, 2030),
        ),
        None,
        "You asked for 2015. The data covers 2010 to 2030 but holds no values for that year.",
    ),
    # A no-data answer whose series states no span.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="2020 to 2025",
            requested_years=(2020, 2025),
        ),
        None,
        "You asked for 2020 to 2025. The data holds no values for those years.",
    ),
    # A unit request whose exchange rates start after the data, on a resubmit.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            shown="1980 to 2024",
            data_years=(1976, 2024),
            converted_years=(1980, 2024),
            limited_by="exchange_rates",
        ),
        "aid",
        "Exchange rates start in 1980, so this shows 1980 to 2024.",
    ),
    # The same over a range the request named.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="1970 to 2024",
            shown="1980 to 2024",
            requested_years=(1970, 2024),
            data_years=(1976, 2024),
            converted_years=(1980, 2024),
            limited_by="exchange_rates",
        ),
        "aid",
        "You asked for 1970 to 2024. Exchange rates start in 1980, so this shows 1980 to 2024.",
    ),
    # Deflators that start after the data and end before it.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            shown="1990 to 2020",
            data_years=(1976, 2024),
            converted_years=(1990, 2020),
            limited_by="deflators",
        ),
        "aid",
        "Deflators cover only 1990 to 2020, so this shows 1990 to 2020.",
    ),
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            shown="1976 to 2020",
            data_years=(1976, 2024),
            converted_years=(1976, 2020),
            limited_by="exchange_rates",
        ),
        "aid",
        "Exchange rates end in 2020, so this shows 1976 to 2020.",
    ),
    # Deflators that leave out some donors' rows: before, after and inside the years kept,
    # every year, and a count past the third.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="unit:",
            omitted_years=(
                ("Taiwan", tuple(range(1990, 2023)), (2023, 2024)),
                ("Azerbaijan", (2023, 2024), (2020, 2021, 2022)),
                ("Cyprus", (2001, 2002, 2005), (2000, 2003, 2004, 2006)),
                ("Kuwait", (2010,), ()),
                ("Malta", (1999,), (2000,)),
            ),
        ),
        "aid",
        "Constant prices leave out Taiwan before 2023, Azerbaijan after 2022, Cyprus in 2001 "
        "to 2002 and 2005, and 2 others, which the price deflators do not cover.",
    ),
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="unit:",
            omitted_years=(("Kuwait", (2010,), ()),),
        ),
        "aid",
        "Constant prices leave out Kuwait, which the price deflators do not cover.",
    ),
    # Entities no held name covers join the count, or make it alone.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="unit:",
            omitted_years=(("Kuwait", (2010,), ()),),
            omitted_unnamed=1,
        ),
        "aid",
        "Constant prices leave out Kuwait and 1 other, which the price deflators do not cover.",
    ),
    (
        DepartureFact(code="year_unavailable", binding_ref="unit:", omitted_unnamed=2),
        "aid",
        "Constant prices leave out some years of 2 entities, which the price deflators do not "
        "cover.",
    ),
    # A conversion the loaded tables cannot read names no entity.
    (
        DepartureFact(code="year_unavailable", binding_ref="unit:", omission_undecided=True),
        "aid",
        "Constant prices may leave out years the price deflators do not cover.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Somaliland",
            entities=("Somaliland",),
        ),
        None,
        "You asked for Somaliland. The data holds no values for it.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="origin:",
            asked="Gates Foundation, Wellcome Trust",
            entities=("Gates Foundation", "Wellcome Trust"),
        ),
        "aid",
        "You asked for Gates Foundation and Wellcome Trust. The data holds no values for them.",
    ),
    # A latest request names the latest period the answer serves.
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Ghana",
            entities=("Ghana",),
            latest_years=(2024, 2024),
        ),
        None,
        "You asked for Ghana. The data holds no values for it in 2024.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Ghana, Togo",
            entities=("Ghana", "Togo"),
            latest_years=(2019, 2024),
        ),
        None,
        "You asked for Ghana and Togo. The data holds no values for them in the latest period.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="origin:",
            asked="EU institutions",
            entities=("Bulgaria", "Cyprus", "Latvia"),
            group="European Union",
            group_size=27,
            latest_years=(2024, 2024),
        ),
        "aid",
        "The data holds no values in 2024 for 3 of the 27 European Union members.",
    ),
    # crs-006: the request named the group, so the text counts its members and names none.
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="origin:",
            asked="EU institutions",
            entities=("Bulgaria", "Cyprus", "Latvia"),
            group="European Union",
            group_size=27,
        ),
        "aid",
        "The data holds no values for 3 of the 27 European Union members.",
    ),
    # A region's places are in it, as the heading reads them.
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Africa",
            entities=("Eritrea", "Somalia"),
            group="Africa",
            group_size=54,
            group_dcid="africa",
        ),
        None,
        "The data holds no values for 2 of the 54 countries in Africa.",
    ),
    # A role binding a region's members beside a place: the region's members are counted
    # and only the place named is asked for.
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Africa",
            sets=(UnservedMembers("Africa", "africa", 54, "Africa", ("Ghana", "Tanzania")),),
        ),
        None,
        "You asked for Africa. The data holds no values for 2 of the 54 countries in Africa.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Norway, Africa, Baltic states",
            entities=("Norway",),
            sets=(
                UnservedMembers("Africa", "africa", 54, "Africa", ("Ghana",)),
                UnservedMembers("Baltic states", None, 3, "Baltic states", ("Latvia",)),
            ),
            latest_years=(2024, 2024),
        ),
        None,
        "You asked for Norway. The data holds no values for it in 2024. The data holds no "
        "values in 2024 for 1 of the 54 countries in Africa and 1 of the 3 Baltic states.",
    ),
    # ids-016: a label ending in a role noun keeps it.
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="origin:",
            asked="G20 creditors",
            entities=("Argentina", "Australia"),
            group="G20 creditors",
            group_size=19,
        ),
        "debt_service",
        "The data holds no values for 2 of the 19 G20 creditors.",
    ),
    # The registry withholds the one source holding values there.
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Kenya",
            entities=("Kenya",),
            sources=("World Bank WDI",),
        ),
        None,
        "You asked for Kenya. The data holds no values for it. Search does not read World "
        "Bank WDI, which holds values for it.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="origin:",
            asked="EU institutions",
            entities=("Bulgaria", "Cyprus"),
            group="European Union",
            group_size=27,
            sources=("Eurostat", "IMF"),
        ),
        "aid",
        "The data holds no values for 2 of the 27 European Union members. Search does not "
        "read Eurostat and IMF, which hold values for them.",
    ),
    # mls-001: every series modelled.
    (
        DepartureFact(code="modelled_values", binding_ref=None, shown="modelled"),
        "aid",
        "These values are modelled estimates.",
    ),
    (
        DepartureFact(code="modelled_values", binding_ref=None, partial=True),
        "aid",
        "Some of these values are modelled estimates.",
    ),
    # crs-006: the group is named once, its members never.
    (
        DepartureFact(code="group_read_as_members", binding_ref="origin:", asked="EU institutions"),
        "aid",
        "You asked for EU institutions. This shows its members, each on its own.",
    ),
    # A region read as members is named as the places in it.
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="where:",
            asked="Africa",
            group_dcid="africa",
        ),
        "aid",
        "You asked for Africa. This shows the countries in it, each on its own.",
    ),
    # A region the membership policy holds reads as the places in it whatever its graph
    # type: dev types the Americas a `GeoRegion`.
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="where:",
            asked="the Americas",
            group_dcid="undata-geo/G00134000",
        ),
        "aid",
        "You asked for the Americas. This shows the countries in it, each on its own.",
    ),
    # The CRS publishes no Africa total, so its members leave out the regional amounts.
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="where:",
            asked="Africa",
            group_dcid="africa",
            excluded=(
                "Africa, unallocated by country (DAC)",
                "North of Sahara, unallocated by country (DAC)",
            ),
        ),
        "aid",
        "You asked for Africa. This shows the countries in it, each on its own, "
        "and leaves out what the data allocates to no single country: Africa, unallocated by "
        "country (DAC); North of Sahara, unallocated by country (DAC).",
    ),
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="origin:",
            asked="the EU",
            group_dcid="EuropeanUnion",
        ),
        "aid",
        "You asked for the EU. This shows its members, each on its own.",
    ),
    # DAC1 publishes the DAC's own figure, which one built from its members can differ from.
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="origin:",
            asked="DAC",
            group_dcid="groups/OECD.DAC",
            sum_differs=True,
        ),
        "aid",
        "You asked for DAC. This shows its members, each on its own, and the source publishes "
        "its own figure for the group, which can differ from one built from its members.",
    ),
    # A resubmit of crs-006 posts no words for the group.
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="origin:",
            group="European Union",
            group_size=27,
        ),
        "aid",
        "This shows the 27 European Union members, each on its own.",
    ),
    (
        DepartureFact(
            code="whole_total_read",
            binding_ref="origin:",
            over="origin",
            group="official donors (DAC)",
            whole_of=("official",),
            side_classes=("official", "private", "residual"),
            inferred=True,
        ),
        "aid",
        "No donor was named, so this shows official donors (DAC), the source's total for "
        "every official donor. Private donors are outside it. To list each donor, search again "
        "with open_roles ['donor'].",
    ),
    # DAC2b publishes no private donor, so its official donors' total leaves none out.
    (
        DepartureFact(
            code="whole_total_read",
            binding_ref="origin:",
            over="origin",
            group="official donors (DAC)",
            whole_of=("official",),
            side_classes=("official", "residual"),
            inferred=True,
        ),
        "aid",
        "No donor was named, so this shows official donors (DAC), the source's total for "
        "every official donor. To list each donor, search again with open_roles ['donor'].",
    ),
    (
        DepartureFact(
            code="whole_total_read",
            binding_ref="origin:",
            asked="all donors including private",
            over="origin",
            group="all donors including private",
            entities=("official donors (DAC)", "private donors (DAC)"),
            whole_of=("official", "private"),
        ),
        "aid",
        "You asked for all donors including private. This shows official donors (DAC) plus "
        "private donors (DAC). To list each donor, search again with open_roles ['donor'] in "
        "place of all donors including private.",
    ),
    (
        DepartureFact(
            code="whole_total_read",
            binding_ref="destination:",
            over="destination",
            group="all recipients (DAC)",
            whole_of=("official", "residual"),
        ),
        "aid",
        "This shows all recipients (DAC), the source's total for every recipient, which "
        "includes aid not allocated to a country. To list each recipient, search again with "
        "open_roles ['recipient'].",
    ),
    # The raw-text door asks for a side's members in the question's words.
    (
        DepartureFact(
            code="whole_total_read",
            binding_ref="origin:",
            over="origin",
            group="official donors (DAC)",
            whole_of=("official",),
            side_classes=("official", "residual"),
            inferred=True,
            members_in_words=True,
        ),
        "aid",
        "No donor was named, so this shows official donors (DAC), the source's total for "
        'every official donor. To list each donor, add "by donor" to the question.',
    ),
    (
        DepartureFact(
            code="whole_total_read",
            binding_ref="destination:",
            asked="all recipients",
            over="destination",
            group="all recipients (DAC)",
            whole_of=("official", "residual"),
            members_in_words=True,
        ),
        "aid",
        "You asked for all recipients. This shows all recipients (DAC), the source's total for "
        "every recipient, which includes aid not allocated to a country. To list each "
        'recipient, write "by recipient" in place of all recipients.',
    ),
    (
        DepartureFact(
            code="whole_total_unpublished",
            binding_ref="origin:",
            asked="all donors",
            shown="OECD Creditor Reporting System (CRS)",
            over="origin",
            whole_of=("official", "private"),
        ),
        "aid",
        "You asked for all donors. OECD Creditor Reporting System (CRS) publishes no total of "
        "every donor, so this covers every donor it holds with data, private donors among them.",
    ),
    (
        DepartureFact(
            code="whole_total_unpublished",
            binding_ref="where:",
            asked="all recipients",
            shown="OECD Creditor Reporting System (CRS)",
            over="destination",
            whole_of=("official", "residual"),
        ),
        "aid",
        "You asked for all recipients. OECD Creditor Reporting System (CRS) publishes no total "
        "of every recipient, so this covers every recipient it holds with data, which includes "
        "aid not allocated to a country.",
    ),
    (
        DepartureFact(
            code="direction_discarded", binding_ref="where:", asked="Kenya", over="destination"
        ),
        None,
        "You named Kenya as a recipient. This measure has no recipients, so this shows the "
        "place itself.",
    ),
    # An undirected place whose data sits under one direction alone.
    (
        DepartureFact(
            code="direction_inferred", binding_ref="where:", asked="Kenya", over="destination"
        ),
        "aid",
        "You named Kenya without a direction. Only the data with Kenya as the recipient "
        "holds values, so this reads it as the recipient.",
    ),
    # crs-010: the check covered only the ODA default the engine chose.
    (
        DepartureFact(code="scope_limited", binding_ref="how:crsTypeOfFlow", limits=("ODA",)),
        "aid",
        "This check covered only ODA.",
    ),
]


_NO_DATA_CASES: list[tuple[DepartureFact, str | None, str]] = [
    # crs-020: the closest measure was checked, not shown.
    (
        DepartureFact(
            code="closest_match",
            binding_ref=None,
            asked="aid",
            shown="official development assistance grants for poverty reduction",
        ),
        "aid",
        "You asked for aid. This checked official development assistance grants for poverty "
        "reduction, the closest measure the data holds.",
    ),
    (
        DepartureFact(code="closest_match", binding_ref=None, asked="aid"),
        "aid",
        "You asked for aid. This checked the closest measure the data holds.",
    ),
    (
        DepartureFact(code="closest_match", binding_ref=None, shown="neonatal mortality rate"),
        None,
        "This checked neonatal mortality rate, the closest measure the data holds.",
    ),
    # weo-041.
    (
        DepartureFact(code="measure_not_applied", binding_ref="how:nettingBasis", asked="gross"),
        None,
        'You asked for "gross". This check left it out.',
    ),
    (
        DepartureFact(
            code="measure_not_applied",
            binding_ref="unit:",
            asked="in constant 2022 dollars",
            shown="current US dollars",
        ),
        "aid",
        "You asked for constant 2022 dollars. This checked current US dollars.",
    ),
    # ids-027.
    (
        DepartureFact(
            code="operation_not_applied", binding_ref=None, asked="top 10 debtors", verb="rank"
        ),
        "debt_stock",
        "You asked for a ranking. Search does not compute rankings.",
    ),
    (
        DepartureFact(
            code="breakdown_not_applied", binding_ref="origin:", asked="by creditor", over="origin"
        ),
        "debt_stock",
        "You asked for a breakdown by creditor. This checked the total; name a creditor to "
        "see one.",
    ),
    # A no-data part shows no years, whatever the fact holds as shown.
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked="2030",
            shown="1973 to 2024",
            requested_years=(2030, 2030),
            data_years=(1973, 2024),
        ),
        "aid",
        "You asked for 2030. The data covers only 1973 to 2024.",
    ),
    (
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            requested_years=(2030, 2030),
        ),
        "aid",
        "The data holds no values for the year requested.",
    ),
    (
        DepartureFact(
            code="measure_not_applied",
            binding_ref="how:crsTypeOfFinance",
            values=("debt instruments",),
        ),
        "aid",
        "This check left out debt instruments.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Somaliland",
            entities=("Somaliland",),
        ),
        None,
        "You asked for Somaliland. The data holds no values for it.",
    ),
    (
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Kenya, Uganda",
            entities=("Kenya", "Uganda"),
            sources=("World Bank WDI",),
        ),
        None,
        "You asked for Kenya and Uganda. The data holds no values for them. Search does not "
        "read World Bank WDI, which holds values for them.",
    ),
    (
        DepartureFact(code="modelled_values", binding_ref=None, partial=True),
        "aid",
        "Some of these values are modelled estimates.",
    ),
    (
        DepartureFact(code="group_read_as_members", binding_ref="origin:", asked="EU institutions"),
        "aid",
        "You asked for EU institutions. This checked its members, each on its own.",
    ),
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="origin:",
            group="DAC",
            group_size=32,
            group_dcid="groups/OECD.DAC",
            sum_differs=True,
        ),
        "aid",
        "This checked the 32 DAC members, each on its own, and the source publishes its own "
        "figure for the group, which can differ from one built from its members.",
    ),
    (
        DepartureFact(
            code="group_read_as_members",
            binding_ref="origin:",
            group="European Union",
            group_size=27,
        ),
        "aid",
        "This checked the 27 European Union members, each on its own.",
    ),
    (
        DepartureFact(
            code="whole_total_read",
            binding_ref="origin:",
            over="origin",
            group="official donors (DAC)",
            whole_of=("official",),
            side_classes=("official", "private", "residual"),
            inferred=True,
        ),
        "aid",
        "No donor was named, so this checked official donors (DAC), the source's total for "
        "every official donor. Private donors are outside it. To list each donor, search again "
        "with open_roles ['donor'].",
    ),
    (
        DepartureFact(
            code="whole_total_unpublished",
            binding_ref="origin:",
            asked="all donors",
            shown="OECD Creditor Reporting System (CRS)",
            over="origin",
            whole_of=("official", "private"),
        ),
        "aid",
        "You asked for all donors. OECD Creditor Reporting System (CRS) publishes no total of "
        "every donor, so the closest match covers every donor it holds with data, private "
        "donors among them.",
    ),
    (
        DepartureFact(
            code="direction_discarded", binding_ref="where:", asked="Kenya", over="destination"
        ),
        None,
        "You named Kenya as a recipient, but this measure has no recipients, so this checked "
        "Kenya itself.",
    ),
    (
        DepartureFact(
            code="direction_inferred", binding_ref="where:", asked="Kenya", over="destination"
        ),
        "aid",
        "You named Kenya without a direction. Only the data with Kenya as the recipient "
        "holds values, so this reads it as the recipient.",
    ),
    (
        DepartureFact(code="scope_limited", binding_ref="how:crsTypeOfFlow", limits=("ODA",)),
        "aid",
        "This check covered only ODA.",
    ),
]
"""Each code's text on a no-data part, which shows nothing."""

_ALL_CASES = [
    *((fact, family, True, text) for fact, family, text in _CASES),
    *((fact, family, False, text) for fact, family, text in _NO_DATA_CASES),
]


@pytest.mark.parametrize(("fact", "family", "shows_data", "text"), _ALL_CASES)
def test_departure_text(fact, family, shows_data, text):
    assert departure_text(fact, family=family, shows_data=shows_data) == text


@pytest.mark.parametrize("cases", [_CASES, _NO_DATA_CASES])
def test_cases_cover_every_code(cases):
    assert {fact.code for fact, _, _ in cases} == set(DEPARTURE_ORDER)


@pytest.mark.parametrize(("fact", "family", "shows_data", "_text"), _ALL_CASES)
def test_every_text_speaks_to_the_reader(fact, family, shows_data, _text):
    """In the second person for every code but `modelled_values` and `scope_limited`,
    which state a fact about the values and the check, a group's `entity_unavailable`,
    whose `group_read_as_members` departure states the ask, and any departure with nothing asked, from a door that holds
    no words of the request (a resubmit). A part that shows no data claims to show nothing."""
    text = departure_text(fact, family=family, shows_data=shows_data)

    asks = fact.asked is not None
    states_a_fact = fact.code in ("modelled_values", "scope_limited")
    if not states_a_fact and fact.group_size is None and asks:
        assert re.search(r"\byou\b", text, re.IGNORECASE)
    if not shows_data:
        assert not re.search(r"\bthis shows\b", text, re.IGNORECASE)
    assert "—" not in text
    assert "No year was named" not in text


def test_departures_follow_precedence_and_carry_the_fact():
    direction = DepartureFact(
        code="direction_discarded", binding_ref="where:", asked="Kenya", over="destination"
    )
    breakdown = DepartureFact(
        code="breakdown_not_applied", binding_ref="origin:", asked="by creditor", over="origin"
    )

    departures = departures_from_facts(
        [direction, breakdown, direction], family="debt_stock", shows_data=True
    )

    assert departures == [
        Departure(
            code="breakdown_not_applied",
            binding_ref="origin:",
            asked="by creditor",
            text=departure_text(breakdown, family="debt_stock", shows_data=True),
        ),
        Departure(
            code="direction_discarded",
            binding_ref="where:",
            asked="Kenya",
            text=departure_text(direction, family="debt_stock", shows_data=True),
        ),
    ]


@pytest.mark.parametrize(
    ("fact", "missing"),
    [
        (DepartureFact(code="operation_not_applied", binding_ref=None, asked="share of"), "verb"),
        # A part states a price basis only with the prices its byline shows.
        (
            DepartureFact(code="measure_not_applied", binding_ref="unit:", asked="current prices"),
            "price basis shown",
        ),
        # With nothing asked, the text has only the measure shown to state.
        (DepartureFact(code="closest_match", binding_ref=None), "measure shown"),
    ],
)
def test_a_fact_missing_its_value_raises(fact, missing):
    with pytest.raises(ValueError, match=missing):
        departure_text(fact, family=None, shows_data=True)


@pytest.mark.parametrize(
    ("limits", "text"),
    [
        (
            ["the education sector", "ODA"],
            "This check covered only the education sector and ODA.",
        ),
        (["ODA"], "This check covered only ODA."),
    ],
)
def test_no_data_limit_text_names_what_the_check_covered(limits, text):
    result = no_data_limit_text(limits)

    assert result == text
    assert len(result.split()) <= 20
