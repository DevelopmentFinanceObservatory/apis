"""The catalogue job summarises each five-tuple family and role class of the variables a
build holds once, and serving reads the summary of a family as one row."""

from __future__ import annotations

from origin_catalogue.families import (
    FamilyKey,
    FamilySummary,
    FamilyTables,
    ReleaseFamilies,
    family_tables,
    summarise,
    summarise_table,
)
from origin_catalogue.un_mirror import EMPTY_UN_MIRROR, UnArc, UnMirror, UnVariable
from tests.origin_catalogue.fake_catalogue import un_mirror


def _un_mirror(variables: dict[str, tuple[UnArc, ...]]) -> UnMirror:
    """UN tables holding `variables`, each of series `Debt`."""
    return un_mirror(
        {dcid: UnVariable(series="Debt", arcs=arcs) for dcid, arcs in variables.items()}
    )


_DEBT = FamilyKey("Debt", "amount", "measuredValue", None, None, False, False)


def _variable(
    *,
    name: str | None = None,
    constraints: dict[str, str] | None = None,
    qualifier: UnArc | None = None,
    denominator: UnArc | None = None,
    roles: tuple[str, ...] = (),
    measured: str | None = "amount",
) -> tuple[UnArc, ...]:
    """A variable of population `Debt`, stat type `measuredValue`, measuring `measured`,
    with the arcs given."""
    arcs = [
        UnArc(property="populationType", dcid="Debt", types=("Class",)),
        UnArc(property="statType", dcid="measuredValue", types=("StatType",)),
        *(UnArc(property="observationProperties", dcid=role) for role in roles),
    ]
    if measured is not None:
        arcs.append(UnArc(property="measuredProperty", dcid=measured, types=("Property",)))
    if name is not None:
        arcs.append(UnArc(property="name", value=name))
    for prop, value in (constraints or {}).items():
        arcs.append(UnArc(property="constraintProperties", dcid=prop, types=("Property",)))
        arcs.append(UnArc(property=prop, dcid=value, types=("Enum",)))
    arcs.extend(arc for arc in (qualifier, denominator) if arc is not None)
    return tuple(arcs)


def test_a_family_summary_reads_each_member_as_search_reads_its_arcs() -> None:
    """Members group by five-tuple and role class as search groups them: a qualifier typed
    only ProvisionalNode reads as none, a denominator with no value or one typed only
    ProvisionalNode is unresolved, a missing measured property reads empty, and a variable
    declaring a role is apart. Members and values keep table order, a property every member
    asserts at one value is shared, the total is the first named member declaring no
    constraint, resolved or not, and a variable held empty is in no family."""
    provisional = UnArc(
        property="measurementQualifier", dcid="dangling", types=("ProvisionalNode",)
    )
    families = summarise(
        [
            ("Debt_Unnamed", _variable(qualifier=provisional)),
            ("Debt_Imf", _variable(name="Debt to the IMF", constraints={"creditor": "IMF"})),
            (
                "Debt_UnresolvedCreditor",
                (
                    *_variable(name="Debt to an unresolved creditor"),
                    UnArc(property="constraintProperties", dcid="creditor", types=("Property",)),
                ),
            ),
            ("Debt_Total", _variable(name="Old name", qualifier=provisional)),
            (
                "Debt_Bonds",
                _variable(name="Debt to bondholders", constraints={"creditor": "Bondholders"}),
            ),
            ("Debt_Empty", ()),
            ("Debt_ByCreditor", _variable(roles=("financingProvider",))),
            (
                "Debt_Dangling",
                _variable(
                    denominator=UnArc(
                        property="measurementDenominator",
                        dcid="Count_Missing",
                        types=("ProvisionalNode",),
                    )
                ),
            ),
            ("Debt_Unvalued", _variable(denominator=UnArc(property="measurementDenominator"))),
            ("Debt_Unmeasured", _variable(measured=None)),
        ],
        un_only=False,
    )

    assert families == {
        _DEBT: FamilySummary(
            members=(
                "Debt_Unnamed",
                "Debt_Imf",
                "Debt_UnresolvedCreditor",
                "Debt_Total",
                "Debt_Bonds",
            ),
            values={"creditor": ("IMF", "Bondholders")},
            shared={},
            total="Debt_Total",
            total_name="Old name",
            un_only=False,
        ),
        _DEBT._replace(role_bearing=True): FamilySummary(
            members=("Debt_ByCreditor",),
            values={},
            shared={},
            total=None,
            total_name=None,
            un_only=False,
        ),
        _DEBT._replace(
            measurement_denominator="Count_Missing", denominator_unresolved=True
        ): FamilySummary(
            members=("Debt_Dangling",),
            values={},
            shared={},
            total=None,
            total_name=None,
            un_only=False,
        ),
        _DEBT._replace(denominator_unresolved=True): FamilySummary(
            members=("Debt_Unvalued",),
            values={},
            shared={},
            total=None,
            total_name=None,
            un_only=False,
        ),
        _DEBT._replace(measured_property=""): FamilySummary(
            members=("Debt_Unmeasured",),
            values={},
            shared={},
            total=None,
            total_name=None,
            un_only=False,
        ),
    }


def test_a_property_every_member_asserts_at_one_value_is_shared() -> None:
    (family,) = summarise(
        [
            ("Debt_A", _variable(constraints={"debtor": "Government", "creditor": "IMF"})),
            ("Debt_B", _variable(constraints={"debtor": "Government", "creditor": "Bank"})),
        ],
        un_only=True,
    ).values()

    assert family.shared == {"debtor": "Government"}
    assert family.values == {"debtor": ("Government",), "creditor": ("IMF", "Bank")}
    assert family.un_only


def test_a_family_held_by_both_tables_reads_the_arc_tables_first_and_the_un_ones_with_the_origin_on() -> (
    None
):
    """The release's arc-table members come first, DFO then base, each in dcid order: their
    values lead, their total wins, and a property is shared only when both tables share it at
    one value. With the UN origin off the family is the arc-table part alone, the build's own
    join of its DFO and base tables."""
    un = _un_mirror(
        {
            "undata/x/DEBT": _variable(name="UN debt", constraints={"debtor": "Government"}),
            "undata/x/DEBT_TOTAL": _variable(name="UN total"),
        }
    )
    dfo = {"Debt_Total": _variable(name="Debt")}
    base = {"Debt_Gov": _variable(constraints={"debtor": "Government", "creditor": "IMF"})}
    tables = family_tables(dfo=dfo, base=base, un=un)
    release = ReleaseFamilies(dfo=summarise_table(dfo), base=summarise_table(base))

    assert tables.family(_DEBT, un_system_enabled=True, release=release) == FamilySummary(
        members=("Debt_Total", "Debt_Gov", "undata/x/DEBT", "undata/x/DEBT_TOTAL"),
        values={"debtor": ("Government",), "creditor": ("IMF",)},
        shared={},
        total="Debt_Total",
        total_name="Debt",
        un_only=False,
    )
    assert tables.family(_DEBT, un_system_enabled=False, release=release) == tables.arcs[_DEBT]
    assert (
        tables.family(_DEBT._replace(role_bearing=True), un_system_enabled=True, release=release)
        is None
    )


def test_a_family_only_the_un_tables_hold_is_theirs_and_absent_with_the_origin_off() -> None:
    un = _un_mirror({"undata/x/DEBT": _variable(name="D")})
    tables = family_tables(dfo={}, base={}, un=un)

    assert tables == FamilyTables(
        arcs={},
        un={
            _DEBT: FamilySummary(
                members=("undata/x/DEBT",),
                values={},
                shared={},
                total="undata/x/DEBT",
                total_name="D",
                un_only=True,
            )
        },
    )
    assert tables.family(_DEBT, un_system_enabled=False, release=ReleaseFamilies({}, {})) is None
    assert family_tables(dfo={}, base={}, un=EMPTY_UN_MIRROR) == FamilyTables()
