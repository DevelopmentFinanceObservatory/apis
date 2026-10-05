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
`dfo-spec-kit` does not record facet conventions, so this
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
   Commons, DFO's facets first (`materialise.facets.dfo_precedence`),
2. then the facet holding a point in the window at the most bound places,
   or for a role-bearing StatVar at the most donor and recipient pairs
   (`materialise.facets.MemberProbe.breadth`),
3. then, for a year range, the most window years the facet's span covers
   (`materialise.facets.years_in_window`),
4. then most observations (`obsCount`, summed across every bound entity),
5. then most recent `latest_date`,
6. then `provenance_id`,
7. then `measurementMethod`,
8. then `unit`,
9. then `facet_id`.

One facet group serves a StatVar at every bound place, so the facet that
covers the most places leads: a request for Kenya and Ghana serves the
source both publish over a longer series only Kenya holds, and a place the
selected facet does not cover returns no rows. With one bound place every
candidate covers it, and the next step decides.

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
the year changes. Its stored head records the window it read, and the first
read whose clock ends that window at another year reads the data again
(`datasets_api.currency.clock_moved`). The applied window is judged against
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
specifically because the selection feeds `content_version`'s preimage. An
unstable rule would rotate a materialisation's identity for reasons a
caller could not observe or reproduce.

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
  agreement. A `facet_default_selected` warning always fires, naming the
  axis and the selected ref. `basis` is `deployment_fallback` on the wire
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
  binding is refused with `default_facet_undefined` instead, since
  datasets citation assembly requires a concrete provenance ref.

DFO precedence exists because admission works per place. Where DFO holds
data for a StatVar at a place, the origin catalogue withholds every base
facet there, but a base facet admitted at another bound place still joins
the candidate set, and a fetch by its facet id would read its rows at DFO's
places too. Ranking DFO's facet first whenever DFO has data at any bound
place serves DFO's data there, and a place only the base source covers
reads as a coverage gap. For every other StatVar the candidates come from
one origin and step 1 changes nothing.

Step 1 ranks the candidates left after narrowing by the caller's bound axes,
by the native-currency preference for a current-price currency request, and
by the window. The window step runs first, and `dfo_precedence` reads the
narrowed pool, so a window only a base facet reaches serves that base facet
where a DFO facet would return an empty series. A spec that binds a
provenance, unit or measurement method only a base facet carries, or
requests a currency only a base facet is native in, gets that base facet,
and its rows serve only at places where DFO holds no data, since admission
withholds every base facet where DFO's is present.

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
the members of a group and the members of a collection alike, share one
presence read and one facet read at the union of their scopes, and each is
admitted and fetched at its own places, so a member gets the scope, rows
and `content_version` it gets when minted by itself. A group's table holds
each member's rows at its own places: a member holding only a worldwide
total adds an Earth row beside its companions' country rows, and a member
holding country data adds no Earth row, whatever its companions hold. A
`statvar_set` mint naming several StatVars is a collection.

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

A request that binds no entity refuses only when nothing holds data: a
single StatVar returns `default_facet_undefined` for an unbound axis or
`facet_not_available` for a bound one, and a group returns
`facet_not_available`. When the probe is empty because a UN System Data
Commons facet read failed at one of the places, the request refuses
`upstream_error` (retryable), and when every facet there is withheld by
the origin catalogue it refuses `excluded_by_registry`, as a bound
request does. The UN front door refuses a few countries on every read, so
a place it refused leaves the refusal `default_facet_undefined`, and the
refusal's detail names those places, as the `un_system_unavailable`
warning on a served request does. A failed read of Earth's countries or
of presence, in the probe or in the read that explains a refusal, refuses
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

The content-version preimage still omits a null-ref axis entirely, because
its own contract defines the facets it carries as one entry per axis
"actually resolved," and a null ref there is refused outright. The full
resolution, including the null, is still recorded on the manifest's trace.

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
collection, a rule of its own, since it reconciles several members' own
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
