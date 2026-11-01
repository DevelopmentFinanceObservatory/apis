# The default facet policy

Every materialised series carries three facet axes, `provenance`,
`measurementMethod`, and `unit`, naming which source, method, and unit its
rows come from. A caller may bind any axis explicitly (`scope.facets`). An
axis left unbound resolves through a default policy instead. The graph may
declare its own default policy for a given StatVar and axis
(`GraphReader.default_facet_policy`). None exists
(`default_facet_policies_live: false` on the capabilities document). This
document is the fallback this deployment applies when the graph declares
none. It states the rule and the two enforcement points that implement it.
`spec-kit` does not record facet conventions, so this
document is the policy's authoritative record until it does.

The capabilities document publishes this policy's identity as
`default_facet_policy: {"policy": "dfo/facet-policy/most-observations",
"version": "1"}`.

## The rule

The policy picks **one whole candidate facet**, not one value per axis. All
three axes of the selection are read off that same facet. A rule that
resolved each axis independently could select a combination no single facet
actually carries. The resulting request would ask the source for rows that
satisfy provenance, method, and unit simultaneously when no observation
ever did, and it would silently return nothing.

Given the facets a probe reports for the requested StatVar and bound
entities, narrowed to whichever facets also satisfy every axis the caller
_did_ bind, the policy narrows once more by the requested window. When the
request names a year range, it keeps the candidates whose
`[earliest_date, latest_date]` years overlap that range. If none overlaps,
it keeps them all. A date the probe does not report never excludes a
candidate, so a candidate with no `earliest_date` overlaps when its
`latest_date` reaches the window's start year. `all` and `latest` windows
leave the pool unchanged. The surviving candidates rank in one total order:

1. for a StatVar the origin catalogue lists under both DFO and base Data
   Commons, DFO's facets first when DFO's broadest facet covers at least as
   many bound places, or role pairs, as the broadest base facet
   (`materialise.facets.dfo_precedence`),
2. then the facet holding a point in the window at the most bound places,
   or for a role-bearing StatVar at the most donor and recipient pairs
   (`materialise.facets.MemberProbe.breadth`),
3. then, for a year range, the most window years the facet's span covers
   (`materialise.facets.years_in_window`),
4. then most observations (`obsCount`, summed across every bound entity),
5. then most recent `latest_date`,
6. then, for a request binding one place, the most places the facet holds
   a point at across the whole variable at its default place scope
   (`materialise.facets.MemberProbe.variable_breadth`),
7. then `provenance_id`,
8. then `measurementMethod`,
9. then `unit`,
10. then `facet_id`.

One facet group serves a StatVar at every bound place, so the facet that
covers the most places leads: a request for Kenya and Ghana serves the
source both publish over a longer series only Kenya holds, and a place the
selected facet does not cover returns no rows. With one bound place every
candidate covers it, so count and date decide, and facets that tie on both
go to the one serving the variable most widely, as a request naming many
places would elect: France alone gets the IMF's US-dollar GDP (197 places)
over its euro series (25). The whole-variable breadth comes from the
variable's facet profile (`materialise.facet_profile`), read only when the
leading facets tie on every earlier step and differ in provenance, method,
unit or the cadence the dispatch serves, and at most once per variable per
request.

When that read fails, the tie cannot be broken on breadth, and the two
callers differ because one stores its election and the other does not. A
Datasets mint, rebuild or operation refuses rather than store the facet the
unit's name would pick. A transient failure (a timeout, a graph read error
or byte cap, a profile flight the executor refused) refuses
`service_unavailable` (503, retryable), and a retry reads again. A variable
whose own profile raised for any other reason is a defect, logged as an
error with the variable, and refuses `internal_error` (500, not retryable),
since a retry would meet it again. Search falls back to the remaining steps,
so the unit's name decides, and marks the answer degraded, so neither the
answer nor an options result built on it is cached. A failed read that no
tie needed changes nothing.

The window step exists for series whose facets cover different years. A
StatVar with recorded values to 2024 (no `measurementMethod`) and a
projection facet for 2025-2032 (`measurementMethod: dcid:Estimated`)
serves the recorded facet with no window or with a 2020-2027 window, since
both facets overlap that range and the recorded one has more observations.
A 2027 window serves the projection, the only facet with a 2027 point.
Without the step, a window the winning facet does not reach returns an
empty series while another facet holds the requested years. The step needs
no extra request, since the probe already reports each candidate's first and
last dates. A selection can therefore change with the window alone, and a
changed selection changes `content_version`.

A year range with no end year means recorded data: the request ends it at
the current year (`materialise.window.election_window`) and both elects and
reads under that range, so a projection facet that starts later does not
lead a "since 2016" request and no later year is returned. A range starting
after the current year asks about the future and keeps its open end, and a
range naming its end year ("2016 to 2030") reaches projections. Search ends
an open-ended answer at the same year, so an answer and a dataset minted from
its Spec select the same facet and serve the same years. The stored identity
keeps its open end, so the selection and the years served can change when
the year changes. Its stored head's dependency fingerprint records the window
it read, and the first read past the recheck interval whose clock ends that
window at another year reads the data again (`materialise.dependencies`). The applied window is judged against
the years the definition states: data that ends before the current year
answers the whole of "since 2016", so the window's origin is `query` and the
request carries no `coverage_below_scope`. A stated start or end year the
data falls short of is a `coverage_clamp`.

`obsCount` alone is a strong discriminator but not a total order. In a
survey of 2,016 StatVar/entity pairs sampled against the dev graph, 95% of
pairs offered exactly one candidate facet, so this ranking decides only the
remaining 5%. Within that multi-candidate slice,
ranking by `obsCount` alone left ties in 11 of 101 pairs, which is why the
rule chains the further tiebreak keys above. `facet_id` closes the order
safely because it identifies the facet's own metadata content rather than
counting per StatVar. The same survey found a majority of observed
`facet_id`s recurring across more than one StatVar, each mapping to
exactly one (`provenanceId`, `measurementMethod`, `unit`,
`observationPeriod`, `scalingFactor`) tuple.

A missing value sorts last on every key above `obsCount`. Every candidate
carries an `obsCount`: a role-bearing StatVar's candidates take theirs from
the coverage index. `facet_id` is unique per candidate, so the order is
total. The same candidate set always
ranks the same way, in any process, on any call. Determinism matters here
because the selection decides which rows the frame holds, and
`content_version` hashes that frame. An unstable rule would rotate a
materialisation's identity for reasons a caller could not observe or
reproduce.

The top-ranked candidate after narrowing is the selection for every unbound
axis. Each reason below describes the narrowed pool, window step included,
so a window that leaves one candidate yields `sole_candidate`:

- **If every narrowed candidate agrees with the top-ranked facet's value on
  an axis**, that value binds, with reason `sole_candidate` (exactly one
  candidate survived narrowing) or `most_observations` (more than one,
  agreeing).
- **If some narrowed candidate carries a different value, or no value at
  all, on an axis**, the top-ranked facet's own value still binds, reason
  `most_observations_over_disagreement`. This is the one place the policy
  makes a real choice among live alternatives rather than confirming
  agreement. The trace's `choices` names the axis and each candidate value
  in rank order, and the version's description record states the choice
  as a `choose_source` step. `basis` is `deployment_fallback` on the wire
  for this outcome, the same as for every other outcome this rule
  produces -- `declared_default` names a graph-declared policy instead,
  never this deployment's own. The distinction between this outcome and the
  agreeing one above lives entirely in `selection_reason` and the warning.
- **If the candidates disagree on an axis and DFO precedence put the
  top-ranked facet first**, its value binds, reason `dfo_precedence`, with
  no warning.
- **If the top-ranked facet itself carries no value on an axis**, that axis
  binds to null, reason `absent_on_selected_facet`, and resolution
  completes normally -- except on the `provenance` axis, where a null
  binding is refused with `default_facet_undefined` instead, since a
  manifest that serves rows cites a concrete provenance ref.
- **If the probe holds candidates and none survives narrowing by the
  request's bound axes**, each unbound axis is refused with
  `default_facet_undefined`. A probe that holds no candidate at all is the
  empty result described under "A request that binds no entity" below and
  selects nothing.

Admission keeps DFO's and base facets alike at every place
(`origin_catalogue.Catalogue.admit_series`), so the election alone picks the
origin and one facet group serves every bound place. DFO's facet leads when
it covers as many places as the best base facet. A narrower DFO facet gives
way: OECD DAC1 holds `Count_Person` at 50 countries and World Development
Indicators at 169, so a population request for every country serves the
World Bank's 169 from one source. A request read at one series, where every
facet covers it, ranks DFO's facet first. For every other StatVar the
candidates come from one origin and step 1 changes nothing.

Step 1 ranks the candidates left after narrowing by the caller's bound axes,
by the native-currency preference for a current-price currency request, and
by the window. The window step runs first, and `dfo_precedence` reads the
narrowed pool, so a window only a base facet reaches serves that base facet
where a DFO facet would return an empty series. A spec that binds a
provenance, unit or measurement method only a base facet carries, or
requests a currency only a base facet is native in, gets that base facet at
every place it holds, DFO's places included. A spec that binds DFO's
provenance gets DFO's facet however few places it covers.

On the role-bearing path a candidate comes from the series coverage
index. `obsCount` is the index's observation count for the bound roles,
and `earliest_date` and `latest_date` are its first and last covered
years, so the window step tests both ends of the covered range.

## A request that binds no entity

A request whose identity binds no entity is probed and fetched over every
entity that holds data, and the probe and the fetch read the same source.

- A role-bearing StatVar (one with donor and recipient roles) reads the
  series coverage index. An unbound role matches every row, so the
  candidates count every donor and recipient pair, and the fetch sends the
  index's exhaustive filter (`CoverageIndex.exhaustive_single_axis_filter`),
  which lists every covered entity on the smaller role axis.
- A subject-only StatVar covers its default place scope: the countries
  under Earth where it holds admitted data, by dcid, and
  Earth alone, the worldwide total, when no country holds data
  (`materialise.place_scope`, the function Search's default scope calls,
  so a Search answer and its dataset cover the same places). Two graph
  reads give it: Earth's countries, then where the StatVar has data among
  those countries and Earth (`read_default_presence`). The presence read
  counts admitted facets only and reads UN System Data Commons variables
  from the UN deployment. The facet probe reads only those places, the
  fetch names exactly those places, and a fetched row serves only where
  the probe admitted its facet at its place, as for a bound request.
  Regional aggregates, sub-national places and, beside country data, the
  worldwide total are outside the scope; a request names them to get
  them.

A StatVar's scope depends on that StatVar alone. StatVars probed together,
the members of a group, share one presence read and one facet read at the
union of their scopes, and each is admitted and fetched at its own places,
so a member gets the scope, rows and `content_version` it gets when minted
by itself. A group's table holds each member's rows at its own places: a
member holding only a worldwide total adds an Earth row beside its
companions' country rows, and a member holding country data adds no Earth
row, whatever its companions hold.

The probe groups requests by declared roles and entity scope
(`materialise.facets.request_scope_key`), and the fetch groups by the
same key plus origin and window, so a role-bearing member and a
subject-only member that both bind no entity are probed and fetched
separately. A place list longer than one request's URL budget carries is
read in slices (`dc_sdmx.place_slices`), one request after another, with
the row guard counting the rows of every slice together.

The places are discovered at each mint and stay outside `spec_id`, which
hashes only the entities the definition binds. `content_version` changes
with them only through the rows they fetch.

When the probe returns no candidate for any member, the places hold no
observations of the definition's variables and the mint serves the empty
result: a 200 manifest with `coverage.rows` at 0, an `empty_result`
warning whose message says so, no facet selection and no cited source.
This holds for a request that binds no entity and for one that names its
places, for a single StatVar and for a group, and whatever facet axes the
definition binds. No facet is selected and no conversion is planned.

Two causes of an empty probe refuse instead, before a version is stored.
When a UN System Data Commons facet read failed at one of the places, or
the UN front door refused a place the definition names, the request
refuses `upstream_error` (retryable). When every facet at the places of
a subject-only member is withheld by the origin catalogue, it
refuses `excluded_by_registry`. Every subject-only member is checked, in
one read for the members that share a list of places, and a check that
fails refuses `upstream_error` (retryable), or `timeout` when the
request's deadline ends it, so an empty result is stored only when the
check completed. The UN front door refuses a few countries
on every read, so for a request that binds no entity a place it refused
leaves the empty result as it is, and the `un_system_unavailable` warning
counts those places, as it does on a request that served rows. A failed read
of Earth's countries or of presence, in the probe or in the read that
explains an empty probe, refuses
the request `upstream_error` (retryable) and stores no version. A listing
of Earth that names no country is such a failure: Earth holds every
country, and an empty listing taken as "no country" would serve the
worldwide total in place of every country's data.

Earth holds about 250 countries, and the facet probe reads at most 200,000
StatVar and place pairs, each member counted at the union of the scopes it
is read with, so a group of several hundred StatVars probes and a larger
one refuses `collection_probe_ceiling`, with a detail that asks for named
places or fewer members. The count needs each member's own scope, so the
check follows the group's one presence read and precedes every facet
batch. A check made before the presence read could count only every
member at every country, and would refuse groups whose own scopes fit.

Size limits apply as they do to a bound request. The coverage index holds
no subject-only series, so admission has no price for one, bound or not.
The request runs inline and the row ceiling refuses an oversized result
`over_ceiling`.

## Why an absent axis is a binding, not an omission

A `measurementMethod` or `unit` value is legitimately absent on a large
share of real series. In the same survey, roughly two in five pairs
carried no `measurementMethod`, and a similar share carried no `unit`, on
every candidate facet available for them. Refusing whenever the top-ranked
facet lacks a value would refuse a large, ordinary slice of traffic.

An absent axis is recorded as a selection whose `ref` is null, not as an
axis left out of the resolved set. The difference has a concrete
consequence. The row filter and the dispatch-candidate filter both treat an
axis present in the resolved set as a constraint every row or candidate
must match. Leaving the axis out entirely would remove that constraint,
letting a different facet's rows, one that does carry a value there, leak
into the same frame alongside the ones that don't. Binding the axis to
null keeps the constraint active. Only rows whose own cell is null on that
axis are kept, matching the absent value on the winning facet exactly.

The full resolution, including the null, is recorded on the manifest's
trace.

## Enforcement points

Two enforcement points implement this policy:

1. **`materialise.facets.select_facet`**, per axis, once
   `GraphReader.default_facet_policy` has already returned no graph-declared
   policy for that axis. Narrows the probed candidates by bound axes and by
   the request's window (`materialise.facets.narrow_to_window`), ranks the
   survivors (`materialise.facets.rank_candidates`), and returns the
   selection this document describes.
2. **`materialise.generation._generate_member`**, which evaluates every axis
   for a member, aggregates any refusals into one error, and, once every
   axis has resolved, builds the row filter and facet-id dispatch set from
   the resolved selections, so the fetched and served rows match the
   selection exactly.

The dispatch set holds every probed facet carrying the selected value on
each axis, which can be more than one facet id. It holds facets of one
observation period only. The three axes leave out `observationPeriod`, and
a source may publish one series at two cadences under the same provenance
and unit: the UN graph's ILO unemployment rates come as an annual (`P1Y`)
and a quarterly (`P3M`) facet for most countries. The election decides the
period (`materialise.facets.dispatch_cadence`): among the facets carrying
every selected value, the one this policy's ranking puts first, under the
same window and the same breadth the election used, gives its cadence, and
only facets of that cadence dispatch. For an unbound request that facet is
the elected one. A group takes the period serving the most members
(below). For a request binding axes the ranking still decides, so a facet
covering more of the bound places leads one holding more points at fewer
of them. Member selection and group selection return the period with the
selections (`materialise.facets.FacetElection`), and the mint, a preview
and Search's served facets
(`materialise.group_facets.group_served`) all read it.

Each member judges the cadence of every facet id it holds
(`materialise.facets.cadences_by_member`). A stated period is the id's own
and holds for every member. A facet id stating no period reads as annual
for a member only when every place's facet that member holds, as the graph
reports it before any merge, dates by bare years: every date of a complete
date list, or its first and last dates when the read lists no complete set
(`dc_graph.client.Facet.reads_annual`). Merging facets of one id, across
places or blocks of one read, keeps that evidence only when every merged
facet has it, and never judges the widened bounds. Otherwise the id is a
cadence of its own, keyed by the id. So one series split across a `P1Y`
facet and an unstated one serves whole, and an unstated facet that one
place dates by month stands alone. Members are judged apart, because one
unstated id can date one variable by year and another by month: the first
member's copy reads as annual and the second's does not, so no cadence
holds both. Two unstated facets dispatch together only when both read as
annual. A coverage-index candidate's dates are coverage's years, whatever
its cadence, so they are no such evidence: it states its period through
the coverage metadata or stands alone.

A group serves one period, because it mints one frame. The election takes
the period at which the most members hold a facet carrying every selected
value. When any member's facet holds a point in a range window, a member
counts only through facets of its own that do, so a member whose copy of a
facet falls outside the window does not count toward that facet's period.
Among periods held by as many members, the one whose facet ranks first
wins, under the group's ranking over the members' matching facets merged
by facet id and each facet's series count across the members
(`materialise.group_facets.group_breadth`), and then the one the earliest
member holds. Every member dispatches only its facets at that period. A
member holding observations at another period alone dispatches nothing,
and the result carries a `coverage_below_scope` warning naming it by ref
in a structured field. The warning counts such members apart from those whose
observations all fall outside a range window. The first report the
requested years at another frequency, so the frame leaves their figures
out. The second report nothing in those years, and raise no warning when
the served dataset reads an absent member as zero
(`code_index.members.absent_member_reads_zero`), since they add nothing to
a sum over the requested years. A group
chunk fetches the union of its members' dispatch sets, and each member
serves only rows from its own (`materialise.columnar.dispatched_rows`).

Members whose copies of an unstated facet both lack annual evidence share
its cadence and both dispatch it, and their rows can still arrive at two
grains. The mint refuses a frame whose periods hold two grains with
`internal_error` (`materialise.errors.MixedGrainError`), checked on each
member's frame and on every group chunk as it closes, so no such frame is
stored. The presentation assembler refuses one too
(`assembler.frame_view`), so `2019` beside `2019-01` is refused rather
than drawn with one grain's points missing. Neither check sees two
cadences written at one grain (a monthly and a quarterly facet, or an
annual facet dated by month), so dispatch is where cadences are kept
apart.

Beyond that, this policy does not choose between cadences: a member serves
the period of the facet that wins the ranking above, annual or not, and a
group the period serving the most members. No axis lets a caller ask for a
cadence.

A stored dataset head holding two cadences rebuilds at its next source
refresh, which its dependency fingerprint compares past the recheck interval
(`DATASETS_HEAD_RECHECK_SECONDS`, default 900), at the next import for a
head keyed on one, or at the maximum-age rebuild
(`DATASETS_HEAD_MAX_AGE_SECONDS`, default 86,400, or
`DATASETS_HEAD_IMPORTED_DATA_MAX_AGE_SECONDS`, default 604,800, for a head
keyed on an import). Presentation
frames recompute their structural view on load, so the assembler refuses a
stored mixed frame on its next read.

## What this policy does not cover

Binding an axis explicitly always overrides the default policy. A caller
that disagrees with a default selection can pin the axis it cares about.
The window step never narrows a pinned axis or a graph-declared default. A
single member, a unit request, and a group all match them before it runs.

The identity grammar has no way to pin an axis to "absent" the way it can
pin a concrete ref. A caller cannot write `scope.facets` to lock in the
`absent_on_selected_facet` outcome and guard it against a future graph
re-import that adds a value where none existed before. Pinning the axis to
a concrete ref instead locks the result down, but resolves a different, and
possibly narrower, set of rows.

Group materialisation selects one facet across every member of a
group, a rule of its own, since it reconciles several members' own
candidate sets before a single selection can be made. It applies the same
window step to the merged shared candidates before ranking them in the same
order, counting the places each facet covers across the members
(`materialise.group_facets.select_group_facets`, `group_breadth`). The group selection is
drawn from the members whose own probe returns a candidate: a member whose
bound entities publish no observation for its StatVar contributes no
candidate, dispatches nothing, and reaches the frame carrying zero rows,
the same outcome it reaches on its own. A member that has candidates and
shares none of them with the group, or a group where no member has a
candidate at all, stays a refusal.
