The Search engine resolves natural-language queries, structured parameters, and resubmitted specifications into grounded dataset specifications (`spec_id`). It integrates Data Commons knowledge graphs, Gemini structured extraction, SDMX availability checks, and the `materialise` runtime to produce typed dataset coordinates and inline data previews.

## Architecture

The resolution lifecycle processes queries through six sequential stages:

```
Query -> Extract -> Recall -> Shape -> Bind -> Materialise -> Ground -> Spec
```

1. **Extract**: Parses natural-language input into structured components, identifying target variable phrases, geographic entities, temporal constraints, and role axes.
2. **Recall**: Queries the indicator resolver on our Data Commons and on the UN System Data Commons in parallel, keeps the candidates the origin catalogue admits, and ranks each Data Commons' candidates by cosine similarity score.
3. **Shape**: Maps recalled variables into structural sentence shapes, defining required slots across dimensions (`what`, `how`, `where`, `when`, `source`).
4. **Bind**: Grounds identified entities, geographic areas, dates, units, and facet coordinates against knowledge graph nodes and SDMX dimensions.
5. **Materialise**: Evaluates observation availability across candidate slices and selects valid series coordinates using `materialise` graph and SDMX readers.
6. **Ground**: Assembles the canonical `Spec` definition, computes the deterministic content hash (`spec_id`), and returns the typed `ResolveResponse`.

Recall runs once per variable, and Shape through Ground run in two passes (`region/pipeline.py`). The first pass uses our Data Commons' candidates. When it ends `no_data` for a reason other than the request's entities (`entity_not_resolved`, `entity_ambiguous`, `entity_role_not_admitted`), the second pass runs over the UN System Data Commons' candidates with the same resolved entities and deadline. Its answer stands when it is `definite` or `candidates`; otherwise the first pass's `no_data` stands.

Every provenance on the wire names its source and the Data Commons its data comes from (`dfo`, `base` or `un_system`), read with its label from the request's origin catalogue (a provenance the catalogue does not list keeps its graph label and has no source or Data Commons): `resolved.facets.provenance`, each series' `facet.provenance` and `source_refs`, and `trace.resolved_sources`. Each shape lists the Data Commons holding data for its members in `data_commons`, and a shape whose members all come through one upstream source names it in `upstream`. SDMX serves DFO's role-bearing variables only, so the role-series read raises `UpstreamVariableRoutedToSdmx` before it sends a variable the catalogue places upstream.

## Submodules

The search package comprises three submodules:

### `engine`

The core orchestration runtime for resolution and HTTP surface mounting.

- **`core.py`**: Coordinates the end-to-end resolution flow. `resolve_events` provides asynchronous event generation for streaming responses; `resolve_async` provides buffered JSON resolution.
- **`region/`**: Manages per-variable resolution (`region/pipeline.py`), candidate disambiguation, near-neighbour score margin analysis, and facet compatibility verification.
- **`retrieval/`**: Interfaces with the Data Commons knowledge graph to discover statistical variables, expand topics, and verify series coverage.
- **`clients/`**: Encapsulates upstream communication with Data Commons graph APIs (`graph.py`'s contract, `graph_client.py`'s implementation over the shared `dc_graph.GraphClient`), SDMX endpoints (`sdmx.py`), Google Gemini models (`llm.py`), and the `materialise` subsystem (`materialise_bridge.py`).
- **`options/`**: Implements candidate enumeration for entity and facet roles, allowing clients to query valid peer and child selections on demand.

### `models`

Pydantic models defining the search contract and wire representations:

- Status literals: `definite`, `candidates`, `no_data`, `unsupported`.
- Grammar slot axes (`what`, `how`, `where`, `when`, `source`) and facet axes (`provenance`, `measurementMethod`, `unit`).
- Provenance refs (`ProvenanceRef`, with `SourceRef` and `DataCommonsRef`) and the shape's `ShapeUpstream`.
- Request envelopes (`ResolveRequest`, `OptionsRequest`) and response envelopes (`ResolveResponse`, `OptionsResponse`).

### `eval`

The eval case model, the checks that score a response against it, and the paired gate. Cases live in `corpus/*.yaml`; `corpus/README.md` describes the format.

- **`cases.py`**: The `Case` model, the loader that expands measure families, the eligibility rules, and `expectation_hash`.
- **`checks/`**: One module per check (`disposition`, `measure`, `entities`, `provenance`, `request`, `operations`, `refusal`, `warnings`, `candidates`, `parts`, `skills`, `entry_path`, `structural`, `groundedness`, `materialisation`, `task_outcome`). `score` runs them over one raw resolve response and returns a verdict (`pass`, `fail`, `unverifiable` or `n/a`) per check.
- **`runner.py`**: Posts each case to a running app's `/api/search/resolve`, retries infrastructure failures and degraded answers, and picks the modal sample of several by vote.
- **`gate.py`**: The paired regression rule that turns per-sample verdicts on head and base into one verdict per case and check.
- **`summary.py`**: Renders the driver's report as Markdown or terminal text.
- **`live_clients.py`**, **`graph.py`**: The graph, SDMX and coverage readers the groundedness and materialisation checks use.
- **`dataset.py`**, **`sync.py`**: Build a case's request identity, and sync cases to Langfuse datasets.

## HTTP Endpoints

The search surface mounts under `/api/search`.

Every response, success or refusal, carries `x-dfo-trace-id`, `x-dfo-commit`, and a `Server-Timing` header carrying `total`. `diagnostics.trace_id` and `diagnostics.engine_build` in a success body match the header pair, and `diagnostics.timing_ms.total` matches `Server-Timing`'s own `total`, except on a streamed response, where headers go out before the stream and the terminal `done` event's `elapsed_ms` carries the total instead.

### `POST /api/search/resolve`

Resolves a natural language query, structured parse, or base specification resubmission into a dataset specification.

Requests with `Accept: text/event-stream` return a Server-Sent Events (SSE) stream emitting pipeline step events, candidate discoveries, and the final response envelope. Other requests receive buffered JSON responses.

`response_options.include_observations` is off by default because it triggers a per-spec data fetch. Set it to `true` after the engine has selected a definite specification. Buffered JSON embeds the preview in `resolved.observations`; SSE emits it in a trailing `observations` event after `response`. A multi-candidate response retains each candidate's metadata and coverage but does not fetch observation previews. A candidates part carries `observations_omitted` on both transports. Under a converting unit request every preview converts its points to the unit asked, including the bulk previews: one latest point per place for an answer over a defaulted place set, and the first series in identity order for a role-bearing answer over the pair cap. Each converted series carries its `conversion`. A bulk preview leaves out a measure whose points do not convert and names it in an `observations_converted_scope_unsupported` warning, or omits the preview with that reason when no measure converts. A read that fails or crosses the deadline while a bulk preview converts omits the preview with `observations_bulk_fetch_failed` or `observations_timed_out`. When a preview fetch fails, the omission names the read that failed (`graph`, `sdmx` or `upstream response`). An exact `resolved.coverage` carries `evidence`: `facet_metadata` means `observation_count` comes from the graph's facet index and no series ships in that payload, and `observed` means a preview series ships beside it, including a partial delivery where some requested series failed and an `observations_probe_failed` warning names the read that failed. The count is the facet-index count either way. The SSE `response` event always reads `facet_metadata`, because it precedes the fetch.

A role-bearing spec (one whose StatVars declare a donor/recipient flow, e.g. CRS financing) carries an exact `resolved.headline`: the observation, series, and per-role dcid counts in the latest year any of its served series has data, the same counts for that year and the two calendar years before it (`recent_years`, newest first, a year with no data kept at zero so that a latest year far below the ones before it reads as incomplete, and stopping at the first year the coverage artifact represents), whether the request itself named that year, and a sibling `resolved.coverage_stamp` naming the coverage artifact version those counts were confirmed against. Both fields are computed from the published coverage artifact (an in-memory lookup, not a graph or SDMX read), so they cost nothing extra regardless of how broad the answer is; a query that names no donor or recipient at all -- the widest unbound CRS ask -- still resolves `definite` with an exact headline rather than refusing `scope_too_broad`, the same way a narrower, named-role ask does. Both are `null` on a non-role-bearing spec.

#### The `parts` response shape

`ResolveResponse` is a union discriminated on `kind`. A `resolution` carries one entry in `parts` per variable the query asked about, in query order; a `catalogue` answer carries `scope`, and a `declined` answer carries `reason` and `scope`, with no parts. Each part is a complete outcome: it carries its own `status` (`definite`, `candidates`, `no_data`, or `unsupported`), under that status exactly the payload a single-variable response carries (`spec`, `candidates`, `no_data`, or `unsupported`), and the `operations` the question asks for over its data. The resolution's own `status` reads the parts' shared status when every part agrees, or `mixed` when two or more parts differ. A single-variable query returns one part, so `parts[0].status` equals the resolution's `status`.

A raw-text question that extract classifies as `catalogue`, `out_of_scope` or `requires_analysis` returns straight after extract, with no recall, bind or second cache lookup; `out_of_scope` and `requires_analysis` answer `declined` with that `reason`. `scope.place` is the one country the entity resolver confirms among the places the query names, read in one resolver call under the request deadline. No place, a city, a region, an organisation, two countries or more places than a resolution may name leave it null, and so does a resolver failure or timeout, which also keeps the answer out of the response cache. The stream carries `step(extract)`, `response` and `done`, with no `interpretation`, `shape` or `sentence` event. The first cache layer stores these answers, so a repeated "what data do you have?" is served from it. A question of any of the three kinds that names a measure resolves, so text in the query cannot decline a data question, and a resolution with no measure answers `no_data` with `variable_not_resolved`. The `parsed` and `dataset_resubmit` doors run no extract and always answer a resolution.

Two parts that resolve to the same five-tuple shape (`population` and `total population`, say) collapse onto one spec: both parts carry the same `spec_id`, and every part but the first carries a part-scoped `binding_folded` warning naming the fold. A candidates part carries `observations_omitted` when the request set `include_observations`, since a candidates part has not committed to one spec to fetch observations against; `max_candidates` applies per part.

A warning that concerns the whole request (`variables_clamped`, a dropped query qualifier) sits on `diagnostics.warnings`. A warning that concerns one part (`binding_folded`, `observations_omitted`, or that part's own resolved warnings) sits on that part's own `warnings` list.

Every part also carries `restatement`: a `ParsedPart` reproducing that part's own resolution inputs (entities, resolved time window, unit text, source text, operations), as if it had been posted fresh to the `parsed` door. The reproduction is canonical rather than verbatim: a raw_text part's `time_text` is the phrase the engine's own time grammar re-parses back into the identical window, which can differ from the user's original wording. The guarantee covers the resolution inputs and `spec_id`, not the wording. It is null on a `dataset_resubmit` entry's `identity`/`candidates`/`placeholder` kinds, which post an identity or stub rather than a phrase; a `dataset_resubmit` `parsed` entry carries a real one, like any parsed-door part. A leg bound on a directional role axis (origin/destination) echoes that axis, per entity, as the `role_text` a client can post straight back: `from` for the origin axis and `to` for the destination one, chosen from the same lexicon `role_axis_for_text` reads. It is null when the leg's direction came from its own wording alone and no door-supplied role named the axis, when a single name was recorded on both axes and so cannot be reproduced by one role_text, and when one of the leg's own inputs sits outside `ParsedPart`'s field bounds. Every null `restatement` is accompanied by `restatement_reason`, present exactly then, naming which of those cases (or `no_phrase_inputs`, for a part never resolved from a phrase) applies. The withholding applies to re-parsing an edited phrase. For a definite part, the client still holds `spec.definition.identity` and can edit that entity's role directly through a `dataset_resubmit` `identity` entry.

A part's `operations` are the things the question asks done with its data that search does not do, such as a ranking or an explanation. On the raw-text door extract assigns each operation to one variable, so only that variable's part carries it. Its phrase goes to that variable's bind in the same block as the dropped qualifiers, so "top 5 donors for education aid" lets bind leave the donor axis open and the part covers every donor. An operation raises no `query_qualifier_dropped` warning and adds no note to the sentence. The parsed and `dataset_resubmit` doors echo the operations each posted part carries. A request part carries at most 8, the most extract keeps for a whole query.

Every `no_data` part carries `no_data.message`, the plain-language sentence for its `reason` with the indicator, place or category wording the reason names, whether or not `response_options.include_sentence` is set, so a client can show a person the reason beside its code.

An `entity_ambiguous` part carries `no_data.data_query`, a one-part `ParsedInput` built from the part's parsed inputs whether or not it carries a restatement: every entity that resolved to one node is pinned to its dcid, and each ambiguous wording is left unpinned. A client pins the candidate the user picks from `no_data.ambiguous_entities` and posts the part back through the `parsed` door or as a `dataset_resubmit` `parsed` entry. The part's sentence plan marks each ambiguous wording as a `slot` token carrying the role the query gave it, so the client hangs the candidate menu on that phrase.

Timing splits the same way. Each part's own `timing_ms.by_step` names the phases that variable's own leg ran (`recall`, `shape`, `bind`, `materialise`, `answer`), and `total` sums them. `diagnostics.timing_ms` carries the whole request's wall-clock `total` plus `by_step: {"extract": ...}` for extraction, the one step every part shares.

"population and GDP in Brazil" resolves `population` and leaves `GDP` ambiguous:

```json
{
  "schema_version": 1,
  "status": "mixed",
  "query_echo": {
    "entry_path": "raw_text",
    "raw_query": "population and GDP in Brazil",
    "extract_skipped": false
  },
  "diagnostics": {
    "engine_build": "...",
    "trace_id": "...",
    "warnings": [],
    "timing_ms": { "total": 2870, "by_step": { "extract": 340 } }
  },
  "parts": [
    {
      "variable_text": "population",
      "status": "definite",
      "timing_ms": {
        "total": 1180,
        "by_step": {
          "recall": 90,
          "shape": 60,
          "bind": 420,
          "materialise": 610
        }
      },
      "spec": {
        "...": "the grounded Count_Person spec; resolved.observations populated"
      }
    },
    {
      "variable_text": "GDP",
      "status": "candidates",
      "warnings": [
        {
          "code": "observations_omitted",
          "severity": "warn",
          "message": "Observations were not fetched for this candidates part; resubmit a chosen candidate with include_observations to retrieve them."
        }
      ],
      "candidates": {
        "...": "2+ competing GDP specs, none carrying resolved.observations"
      }
    }
  ]
}
```

Each spec's own `resolved.rendered_sentence` is a heading and a byline. The heading states what the answer is, in possessive order with one slot token per axis: "Germany's gross ODA disbursements to Kenya, 1973 to 2024". A unit that changes the magnitude and is no currency (per capita, percent of GDP, PPP international dollars) stays in the heading. The byline carries the latest-year label when one applies, the currency with its price basis ("current US dollars", "constant 2024 euros", "current Guatemalan quetzales", `search.heading.currency_nouns`) as a token with axis `unit`, and the source. The measure noun names no price basis, and the nominal and real siblings are the unit token's options. Every token carries `segment` (`heading`, `byline`, or `departure`). A role with more than three entities reads as its count ("162 recipients"), and a role the request named as a group reads as that group ("the 27 EU member states"). The group rides on `definition.identity.entity_groups`, outside the spec_id hash, so it survives a resubmit. How the answer differs from the request lives on the part, in `departures`: each carries a closed `code`, the `binding_ref` of the token whose edit clears it, what was `asked`, what is `shown`, and a `text` of one or two short sentences ("You asked for 2030. The data ends in 2024, so this shows 1973 to 2024."). On a no-data part the text claims nothing is shown ("You asked for a ranking. Search does not compute rankings."). A named entity the series hold no values for is an `entity_unavailable` departure on its role's token ("You asked for Somaliland. The data holds no values for it."); a role named as a group counts its members without values and names none of them ("The data holds no values for 3 of the 27 European Union members."). A dropped phrase departs unless it is operation wording ("Trend in") or the part's heading and byline already state every word of it, and a dropped direction departs only when the request's own words gave it. A no-data answer whose check covered values the engine chose carries `scope_limited` ("This check covered only ODA."). `binding_ref` names the identity binding whose edit clears the departure; a client links it to the token with the same ref when the plan has one. Departures run in precedence order, from `closest_match` to `scope_limited`, and stay out of the spec sentence, which dfo-ai uses as the spec's label.

Each heading and byline slot token with text carries `states`, the refs a client edits from it: its own `binding_ref`, then every binding the renderer left silent because the token's text already says it. "Gross ODA disbursements" states `how:crsTypeOfFlow`, `how:financialFlowType` and `how:nettingBasis`; "Current US dollars" states `unit:` and `axis:measurementQualifier`, and a heading unit token that names the unit ("in percent") states `unit:`. The source token states `source:`, which is also its `binding_ref` where the answer binds a source slot, so a departure on the dataset links to it either way. When the identity binds `measurementMethod`, a method token with axis `method` follows the source and states `method:`, its `binding_ref`: the method's label ("World Bank Subnational Population Estimates"), "No stated method" for a null ref, or the set's members joined. Method labels drop a trailing full stop where they enter the API (`search.observations.facet_value_label`), so facet options, series facets and the byline read alike. While no method is bound no token states `method:`, and a client offers the method apart from the sentence. The refs come from the same objects that silence a slot (`render.heading.MeasureNoun.states`, `render.how.HowValue.named_by_noun`, `render.heading.Byline.unit_states`), so the two cannot drift. Within a part no ref is stated twice, and a bound slot, an unbound slot with options, or a role that no token states is edited apart from the sentence (`tests/render/test_sentence_golden.py`).

`resolved.unit_conversion` lists what a unit request can convert the answer to (`search.engine.unit.conversion`). Each entry of `currencies` carries the ISO code, the noun the byline names the currency by ("Guatemalan quetzales"), the run of served years a current-price and a constant-price request convert, and the first and last base years a constant-price request converts at. Those years and base years are the ones the conversion accepts (`ConversionAvailability`). A row reads the terms `materialise.convert.row_terms` lists: its native currency's rate in its year, and at current prices its target's rate in that year, or at constant prices its target's rate in the base year and the deflator of its entity on the recipe's deflator role in its year and the base year, the recipe's fallback entity standing in for an entity the table holds no row for. The native currencies are those of the facets the answer's read plan serves, so an answer bound to current US dollars served natively keeps the dollar's rates in later offers. The deflator entities are every entity the plan serves on the deflator role, bound or open, read from the plan and the coverage index in memory. A missing rate refuses the conversion, so the rates bound every row's years. A missing deflator leaves out only its entity's rows, so the constant-price years and the base years run wherever at least one served entity's deflator holds them, and a donor whose deflator starts late shortens no other donor's run. The period a unit request shows (`convertibility.unit_request_period`) reads the same availability. Beside the list sit the default base year and the national currency of each role bound to one entity. The offer is null for a stock, a series not in nominal prices, a unit that is no currency, a provenance with no recipe, or tables that convert no served year. Every fact comes from the conversion catalogue the app loads at startup and reloads when the graph generation changes (`search.engine.unit.conversion_catalogue`), so an answer makes no read for it. Startup waits at most 15 s for the first load and then serves while the load continues in the background. A failed load logs an ERROR and retries after 5, 15 and 60 s, then every 300 s. A reload requested during a load runs once that load ends, so loads run one at a time. While no catalogue is loaded, a currency answer carries the `unit_conversion_unavailable` warning, so a client can tell an outage from an answer that cannot convert.

A unit request whose tables cover only part of the served years is answered over the longest run of years every needed table covers. The part carries a `date_coverage_partial` warning and a `year_unavailable` departure on `when:`: "Exchange rates start in 1980, so this shows 1980 to 2024." The offer's year runs state that period before the reader picks the currency. Inside that period a constant-price conversion leaves out each entity's rows in the years its deflator lacks, and every row of an entity whose deflator lacks the base year (`materialise.convert`). The mint's manifest and the answer carry a `deflator_missing_omitted` warning, and the part a `year_unavailable` departure on `unit:` naming the entities that lose the most years, at most three, then a count: "Constant prices leave out Taiwan before 2023, Azerbaijan before 2022, and Cyprus before 2018, which the price deflators do not cover." (`convertibility.deflator_omission_departure`). Names come from `ConversionCatalogue.entity_names`, which holds every entity a deflator table holds and every place with a national currency; an entity outside both, such as a multilateral donor, joins the count ("and 2 others", or "some years of 2 entities" when none is named). When the loaded tables cannot say whose rows the conversion leaves out (`counted_conversion` is `UNDECIDED`: no catalogue, served facets under several recipes, or no single deflator role), a constant-price answer still carries the warning and a departure naming no entity: "Constant prices may leave out years the price deflators do not cover." A row whose deflator-role cell is null names no entity, so its conversion refuses `rate_coverage_gap` with cause `deflator_missing`, as does a request left with no row.

Every slot menu on a definite answer is counted in memory before the Spec takes it (`search.engine.region.option_coverage.count_slot_options`). `SlotOption.served_rows` is the number of rows a pick's file would hold, in the same terms as `download.rows`: one per series and year inside the requested window, or under `latest` one per series holding its StatVar's latest year. Both read one code path, `CoverageIndex.series_years`, which ORs each series' year masks across the facets that serve it; the download sums the popcounts, and an option sums them over the series its pick reads. A sibling measure in `alternative_measures` carries the same count. `served_rows_exact` says whether the count equals the pick's own `download.rows`. It is false when a counted StatVar's series sit in more than one facet group (provenance, method and unit) in the window, since a per-series read elects one group per StatVar; when the answer reads in bulk and the pick's StatVars sit in more than one facet, since a bulk read elects one facet; under `latest`, when a StatVar has a facet observed more often than yearly, since the coverage index holds years and the read keeps the latest period; when the answer's identity binds a facet value (a unit, a provenance or a method), which the count does not apply, and for a probed option whose facets carry no complete date list. A unit request is applied: each counted series keeps the years its conversion keeps (`search.engine.unit.conversion.CountedConversion`), the years its entity's deflator holds at constant prices and none for an entity no currency names under a conversion into each entity's own currency, as `download.rows` counts them. The count is an upper bound under a unit request only when the loaded conversion tables cannot decide it (no catalogue, several recipes, a unit that is no currency, no single deflator or target role), when a counted series sits in a facet of a currency the answer does not serve, or when the coverage probe counts the menu. `download.rows_exact` and both `served_rows_exact` fields take their exactness from one rule (`counted_exact`). A same-shape merge ORs the masks of a series both answers serve. A StatVar the coverage artifact does not hold adds no rows, so the count covers the pick's other StatVars, and an option all of whose StatVars are missing from the artifact is omitted. A menu omits every option that would serve nothing except the slot's current value, so every option offered has data. The coverage probe under `include_option_coverage` runs only on menus the count leaves open: bound slots of graph-walked families, bound slots of subject-place families, and a menu whose count finds no series for any option while its members carry them. That last case means the coverage index and the members disagree, so it logs a WARNING and the probe counts the menu. The probe counts each option's plan as the download counts a standard plan (`download.pair_extent`).

With `include_sentence` set, the envelope itself carries a composed `rendered_sentence`/`sentence_plan` pair alongside `parts`, describing the whole query in one sentence. When every part is definite, the parts' headings are joined into one sentence. A qualifier (the counterpart, the subject place, the period, the source) that every part states once and alike, matched on text, connective, `binding_ref`, role, and `entity_refs`, is stated once after the joined phrases, in the family's own wording: "USA's health ODA and Britain's education ODA to Ethiopia." A possessive lead every part shares opens the sentence once ("Brazil's total population and nominal GDP, 1960 to 2024"), and parts that differ only in their lead share one noun ("USA's and Britain's health ODA to Ethiopia"). Entity names keep their case. A later phrase's first word is lowered only when it is plain Title Case. A qualifier that differs between parts stays in its own phrase, unless its connective opens with a comma (a period span), which would read as a list item inside the join. In that case, and when two parts' nouns are equal, each heading stays a sentence of its own, a heading two parts state alike is stated once, and the byline they share follows once: "Germany's gross ODA disbursements to Kenya, 2015 to 2020. Germany's gross ODA disbursements to Ghana, 2015 to 2020. Current US dollars · OECD Creditor Reporting System (CRS)." Bylines whose text differs, or any part that is not definite, compose one sentence per part instead, each that part's own sentence followed by its own departures, joined by a plain space: "Brazil's total population, 1960 to 2024. GDP has 2 readings. ...". Beside other parts, a candidates part names its `variable_text` as an editable `what` slot before its own count of readings, an unsupported part follows that slot with its detail, and a no-data part whose reason quotes the variable frames the variable as that slot ("No match for left-handedness. Try a different wording or a related indicator."). Every other no-data sentence names what was checked itself and is mirrored. After the headings comes the byline, once when every part's byline reads the same, then each part's departures in part order, as `segment="departure"` tokens carrying that part's `part_index`. This is the one sentence that writes out a definite part's departures, since a spec's own sentence holds its heading and byline only. Every token in the composed `sentence_plan` carries a `part_index` pointing back into `parts`, or `null` for a token that belongs to the sentence as a whole (an inter-phrase connective, a shared qualifier, lead, or noun, the terminal period, the byline, the space between two per-part sentences). A single-variable query's composed sentence is that part's own sentence followed by its departures. Its `part_index` stamping nulls every qualifier and connective the lone part states, leaving its phrase under its own index.

On the SSE stream, every per-variable event (`shape`, `sentence`, `step`) carries `variable_index`, the same index the terminal `response` event's `parts` list is keyed by. Trailing per-spec events (`slot_options`, `entity_options`, `facet_options`, `observations`) join on `spec_id` instead, since a same-shape collapse can put two parts on the same spec. `sentence` events stay keyed by `variable_index`, one per part; the composed envelope sentence only ever rides the terminal `response` event, alongside the rest of the envelope. A part whose shape carries a measure family bind elected gets a first `sentence` event right after bind, naming only its measured thing from that election ("gross disbursements"), without the flow type and the year, followed by a second `sentence` event once materialise resolves the served rows, which adds both and refines that phrase when the served measure family differs from what bind elected. A part with no elected family (most indicators outside the CRS-style financial-flow axes) gets only the post-materialise `sentence` event, because materialise resolves its measured thing from the served StatVar.

#### Editing an identity

`kind: "dataset_resubmit"` with one `identity` entry posts a resolved identity back for a base-diffed edit. This is the shape a single-identity edit takes: `entries` carries one entry, and the envelope's one part is that identity re-resolved. `identity` is the complete edited definition; `base_spec_id`/`base_identity` name the prior spec it edits against. A no-op resubmit (reposting `base_identity` unchanged as `identity`) returns `base_spec_id` again, because `spec_id` hashes only the identity core (`shape`, `slots`, `entities`, `facets`, `requested_window`). Editing a bound entity role, here `identity.entities.observationAbout`, mints a new `spec_id` bound to the edited place:

```json
{
  "input": {
    "kind": "dataset_resubmit",
    "entries": [
      {
        "entry_kind": "identity",
        "variable_text": "population",
        "identity": {
          "identity_version": "spec1",
          "shape": {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "count",
            "stat_type": "measuredValue"
          },
          "slots": [],
          "entities": { "observationAbout": ["country/DEU"] },
          "requested_window": { "kind": "all" }
        },
        "base_spec_id": "spec_1da23159b03e3b3ca461fa2eaa552fb7",
        "base_identity": {
          "identity_version": "spec1",
          "shape": {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "count",
            "stat_type": "measuredValue"
          },
          "slots": [],
          "entities": { "observationAbout": ["country/FRA"] },
          "requested_window": { "kind": "all" }
        },
        "resolved_stat_var_dcids": ["Count_Person"]
      }
    ]
  }
}
```

`base_identity` is `base_spec_id`'s own identity, bound to France; `identity` is that same document with `observationAbout` changed to Germany. Two facts follow from `spec_id` hashing the identity alone: reposting `base_identity` verbatim as `identity` returns `spec_1da23159b03e3b3ca461fa2eaa552fb7` again, and this edited body returns a different spec_id, `spec_ecb7cbe283c40ecbf80cbf045c934abd`, bound to Germany.

#### Editing a whole result with `dataset_resubmit`

`kind: "dataset_resubmit"` posts every part of a multi-variable result as one input and answers with the same N-part envelope, composed and folded across all N, so editing one part keeps the others' composed sentence and fold in sync. `entries` carries one entry per part, in response part order, each tagged by `entry_kind`. An `identity` entry carries the identity, base_spec_id, base_identity, removed and resolved_stat_var_dcids fields, plus a `variable_text` echo. A `candidates` entry posts a candidate set as a list of identities and gets them back re-materialised as a `candidates` part in the posted order, always, since that order records either an LLM's own ranking or a retrieval-score band, neither of which can be recomputed from the wire. A `placeholder` entry restates a still-unresolved sibling (`no_data`/`unsupported`) as a stub, scalars only. The engine never accepts a `NoData`/`Unsupported` payload back as input: posted state feeds verification, not the response. A `parsed` entry posts one `part` (a `ParsedPart`), ordinarily a prior response part's own `restatement` posted back edited, and is genuinely re-resolved through the parsed door's own per-part path. It is the entry for a sibling whose own text the user changed and which must be re-resolved rather than restated, unlike `placeholder`'s stub. There is no spec registry, so every entry carries its own full identity and the door reads no state from a prior request. `entries` is a full replacement, not a patch: the door is stateless and answers only with the parts it was posted. The response simply comes back with fewer parts when an entry for a part is omitted, rather than a refusal or a preserved copy of the missing part. A malformed entry fails the whole request 4xx, pointer `/input/entries/{i}/...`; a resolution outcome (`no_data`/`unsupported`) is a normal 200 part with siblings unaffected. When several entries are invalid, one rule decides which refusal a client gets. A static check (a placeholder entry's `code` against the code catalog, or a candidates entry's `candidates` length against the deployment's ceiling) always outranks a refusal only a live resolve can discover, and within either class the lowest entry index wins. One refusal therefore always names a single entry to fix, though fixing it can surface a further refusal from another entry on the next request.

### `POST /api/search/options`

Enumerates the candidate value set for one resolved entity role (`observationAbout`, `origin`, or `destination`) of a posted identity, on demand. It carries the same identity core a dataset_resubmit `identity` entry carries (`shape`, `slots`, `entities`, `facets`, `requested_window`), with no `base_spec_id`/`base_identity`/`removed` and no `resolved_stat_var_dcids` seed. The door derives the identity's own StatVar population server-side (`engine.options.population.options_population`): from the packaged code index when it carries the identity's own five-tuple, or from a deterministic graph enumeration (`statvar_members.enumerate_members`) otherwise. It refuses `population_empty` when the identity's shape denotes no population, `scope_too_broad` past the graph enumeration's own `SEARCH_OPTIONS_MEMBER_CAP` member cap, `invalid_role_ref` for a posted entity reference the graph does not carry, `option_target_invalid` for a role the resolved population's own declared role signature does not accept, `observationAbout` against a role-bearing family included, and `exact_binding_unsupported` for a posted slot bound `state="exact"`, which no Search door serves -- only the datasets doors read it. The refusal names the population's own accepted roles as alternatives rather than silently mapping the request onto the subject. `include_coverage` (default false) opts into a per-item observation-coverage probe; the default response carries plain candidate values only. Client interfaces use this endpoint to populate dropdowns and geographic peer selections.

### `GET /api/search/schema`

Returns JSON Schema definitions for `ResolveRequest` and `ResolveResponse`, including references to runtime capability descriptors.

### `GET /api/search/capabilities`

Returns runtime deployment capabilities, including supported recipe classes, active grammar axes, query character limits, and door liveness.

### `GET /api/search/catalogue`

Returns the published search catalogue: `{catalogue_version, counts, examples}`, the counts of indicators, datasets, publishers, countries and aid donors, recipients and years, and the example sentences the catalogue job checked against the engine, each `{text, template, stat_var, place}`. `?place=<country dcid>` returns the counts at that country and the examples whose StatVar has data there, filled with its name. A place that is not a `country/` dcid, or a country the catalogue does not count, refuses `request_invalid` (400). With no catalogue loaded the route answers `service_unavailable` (503) and every other route serves as usual. A 200 carries `Cache-Control: public, max-age=300`. See [deploy/search-catalogue.md](../../deploy/search-catalogue.md).

### `GET /api/search/healthz`

Returns service health status, the running build commit, and the capability version (`{status, build, capability_version}`).

## Configuration

Search behaviour is configured through environment variables.

| Variable                               | Type    | Default                                                          | Description                                                                                                                                                                                                                                             |
| -------------------------------------- | ------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`                       | String  | None                                                             | API key for Gemini structured extraction calls. Required in live deployments.                                                                                                                                                                           |
| `SEARCH_ENGINE_MODEL`                  | String  | `gemini-flash-lite-latest`                                       | Model used for extraction and disambiguation prompts.                                                                                                                                                                                                   |
| `SEARCH_LLM_SEED`                      | Integer | `1`                                                              | Random seed for deterministic LLM sampling.                                                                                                                                                                                                             |
| `SEARCH_GRAPH_BASE`                    | String  | None                                                             | Required. Base URL for the Data Commons knowledge graph service.                                                                                                                                                                                        |
| `SEARCH_GRAPH_TOKEN`                   | String  | Empty                                                            | Shared secret header (`x-graph-token`) for graph WAF bypass.                                                                                                                                                                                            |
| `UN_GRAPH_BASE_URL`                    | String  | `https://unsd-datacommons.gcp.un-icc.cloud/core/api`             | API root of the UN System Data Commons, which serves `undata/` dcids. It takes no token. Unprefixed and shared with every surface and the origin catalogue job.                                                                                         |
| `SEARCH_GRAPH_TIMEOUT_S`               | Float   | `30.0`                                                           | Passed to httpx.Client(timeout=...) as one number, which sets the connect, read, write, and pool timeouts.                                                                                                                                              |
| `SEARCH_UN_RESOLVE_TIMEOUT_S`          | Float   | `3.0`                                                            | Timeout for the UN System Data Commons resolve that starts beside ours in recall. Only a part whose first pass ends `no_data` waits for it. On failure or timeout, search answers from our candidates, and such a part carries `un_system_unavailable`. |
| `SEARCH_RELATED_MEASURES_TIMEOUT_S`    | Float   | `1.0`                                                            | Timeout for a definite answer's related-measures graph reads, which run beside its other post-election reads. Past it, or on a failed read, the Spec states `related_measures` as null.                                                                 |
| `SEARCH_LLM_TIMEOUT_S`                 | Float   | `30.0`                                                           | HTTP timeout in seconds for Gemini API calls.                                                                                                                                                                                                           |
| `SEARCH_LLM_MAX_CONCURRENCY`           | Integer | `32`                                                             | Most Gemini requests one event loop has open at once; excess requests wait up to `SEARCH_LLM_TIMEOUT_S`.                                                                                                                                                |
| `SEARCH_REQUEST_TIMEOUT_S`             | Float   | `45.0`                                                           | Maximum execution deadline for resolution requests.                                                                                                                                                                                                     |
| `SEARCH_MAX_QUERY_CHARS`               | Integer | `2000`                                                           | Maximum character length permitted for input text queries.                                                                                                                                                                                              |
| `SEARCH_SDMX_MAX_ROWS`                 | Integer | `200000`                                                         | Safety row cap applied to broad SDMX data read requests.                                                                                                                                                                                                |
| `SEARCH_OBS_PREVIEW_MAX_ROWS`          | Integer | `20000`                                                          | Safety row cap applied to a single-pair observations-preview fetch.                                                                                                                                                                                     |
| `SEARCH_UNKNOWN_SCORE_THRESHOLD`       | Float   | `0.65`                                                           | Minimum cosine relevance floor required to retain candidate variables.                                                                                                                                                                                  |
| `SEARCH_NEAR_MARGIN`                   | Float   | `0.025`                                                          | Score margin required to distinguish competing candidate variables.                                                                                                                                                                                     |
| `SEARCH_MAX_OPTIONS`                   | Integer | `500`                                                            | Maximum items returned by the options discovery endpoint.                                                                                                                                                                                               |
| `SEARCH_OBS_PREVIEW_POINTS`            | Integer | `20`                                                             | Maximum observations returned per series in inline previews.                                                                                                                                                                                            |
| `SEARCH_OBS_PREVIEW_CONCURRENCY`       | Integer | `8`                                                              | Fetch concurrency bound for one observations-preview call (1 is sequential).                                                                                                                                                                            |
| `SEARCH_OBS_PREVIEW_TOTAL_CONCURRENCY` | Integer | `32`                                                             | Process-wide bound on in-flight observations-preview fetches across all calls.                                                                                                                                                                          |
| `COVERAGE_ARTIFACT_URI`                | String  | None (required)                                                  | Base URI of the published coverage artifact this surface loads at startup; a missing or unreadable artifact fails startup.                                                                                                                              |
| `COVERAGE_RECHECK_SECONDS`             | Float   | `300`                                                            | How often a request-triggered background reload checks for a newer published coverage stamp.                                                                                                                                                            |
| `ORIGIN_CATALOGUE_URI`                 | String  | None (required)                                                  | Base URI of the published origin catalogue this surface loads at startup; a missing or unloadable catalogue fails startup. Each request binds the loaded build.                                                                                         |
| `SEARCH_CATALOGUE_URI`                 | String  | None                                                             | Base URI of the published search catalogue `GET /api/search/catalogue` serves. Unset or unloadable, that route answers 503 and startup is unaffected.                                                                                                   |
| `SEARCH_CATALOGUE_RECHECK_SECONDS`     | Float   | `300`                                                            | How often a request-triggered background reload checks for a newer published search catalogue build.                                                                                                                                                    |

## Testing and Evaluation

### Unit and Integration Tests

Execute the engine test suite with `pytest` or `just`:

```bash
uv run pytest tests/engine/
# or
just test tests/engine/
```

Tests cover:

- Direct pipeline execution and event stream validation
- Facet compatibility and refusal classification
- Near-neighbour candidate score disambiguation
- Options door resolution and surface generation

### Eval cases

Score the eval cases on a running app with the paired-gate driver:

```bash
just eval-app 8100
just eval-run http://127.0.0.1:8100
# or
uv run python scripts/eval_paired.py --single-url http://127.0.0.1:8100
```

The driver posts each eligible case to `/api/search/resolve` and prints every check that did not pass, per case. `just eval-paired` compares a head app against a base app. `scripts/README.md` lists the options.
