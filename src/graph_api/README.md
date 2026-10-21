The Graph API surface reads Data Commons Platform (DCP) graphs and the tables DFO holds beside them. It resolves free text to graph references, describes any node the graph holds under one envelope per kind, lists what belongs to a node (codes, datasets, variables, facets, group and region members), and assembles a per-source documentation document rolled up from a Dataset's own Provenance children.

The surface mounts at `/api/graph` and operates as an independent composition root within `dfo-apis`.

## Architecture

```
 POST /resolve        GET /node/{dcid}      GET /node/{dcid}/members     GET /documentation/{dcid}
                      POST /nodes           GET /roots                   GET /sources
      |                    |                        |                            |
      v                    v                        v                            |
+------------------+  +------------------------------------------+               |
| EntityResolver   |  | In-process tables: code index, origin    |               |
| (resolvekit      |  | catalogue, membership policy, coverage   |               |
| match, gate,     |  | index. The node card also reads a facet  |               |
| confirmation)    |  | profile and the SDMX availability.       |               |
+------------------+  +------------------------------------------+               |
      |                    |                        |                            |
      v                    v                        v                            v
+---------------------------------------------------------------------------------------+
|          GraphNodeReader (DCP) behind DcpReadExecutor: one admitted read per worker   |
+---------------------------------------------------------------------------------------+
```

### Components

- **Entity resolution (`entity_resolution`)**: Free-text resolution and graph confirmation both run inside the shared `entity_resolution.EntityResolver` component (`packages/dfo-data-tools/src/entity_resolution`), acquired once per process (`entity_resolution.acquire_shared()`) and shared with Search when both are mounted in the same deployment. It resolves over `resolvekit`'s four bundled country-level geo modules plus `dfo_graph`, a custom pack built from this deployment's own graph (`entity_resolution.MODULE_IDS`). Both are baked into the image at build time (`scripts/warm_resolvekit_cache.py`), so no pack is built or composed at runtime. The SQLite engine under `resolvekit` is thread-affine, so all matching queries run on the component's own single-worker thread. Graph confirmation runs on this surface's DCP-read executor. The resolve door maps the component's result onto the wire and takes a group's or region's count, link and inline members from the membership policy, with no graph containment read.
- **Membership policy (`entity_resolution.membership.MembershipPolicy`)**: The lifespan publishes the shared resolver's policy as `NODE_MEMBERSHIP_SLOT`. The resolve, node and members doors classify and list groups and regions from it. It holds 48 groups, taken from resolvekit's curated membership (13 of them date their rows, so they accept `as_of`), and 29 regions, taken from resolvekit's UN M49 containment (undated). `data/membership_policy.json` composes North America from M49 Northern America plus Central America and the Caribbean (23 states) and declares which region projections resolvekit holds only in part. A group takes its resolvekit id as its dcid. `entity_resolution/data/group_ids.json` replaces the ids of the groups whose membership has its own graph node: the G7 (`groups/G7`) is `GroupOfSeven`, and the OECD (`groups/OECD`) is `undata-geo/G00406000`. It also maps seven DAC regions to the region whose countries they list ("Africa (DAC)" to `africa`, "Caribbean (DAC)" to `Caribbean`, and Central America, Melanesia, Micronesia, Polynesia and Southern Africa); the policy holds the region alone under that id and refuses to build when the two member lists differ (`build_policy`). Resolvekit's other 46 groups keep their own ids.
- **Kinds (`graph_api.kinds`)**: One list of 16 kinds serves resolve and the node route. `classify` gives a node its kind by a fixed order: membership policy groups, then `typeOf`, then the code index, then membership policy regions, units, aggregates and the resolver's place and organisation types.
- **DCP read executor (`DcpReadExecutor`)**: Graph confirmation reads, the node, members and documentation doors' reads and the variable card's name and currency reads all run on one bounded thread pool (`GRAPH_NODE_POOL_SIZE`). The executor admits one read per worker. A read that arrives while every worker is busy is refused at once as `503 service_unavailable`, which is retryable. A running read keeps its worker until it returns, after its request has timed out. The executor logs one WARNING when its workers fill (`thread pool graph-dcp saturated`) and an INFO line when they drain. Each read logs how long it waited between submission and a worker starting it (`DCP read <function> waited <n> ms for a worker`), and each upstream response logs its method, path, status and milliseconds to headers (`graph POST /core/api/v2/node answered 200 in <n> ms`), one line per attempt. Both lines are DEBUG, on the `graph_api` and `dc_graph` loggers. A wait over 1 s (`graph_api.config.DCP_QUEUE_WAIT_WARN_SECONDS`) or a response over 2 s (`dc_graph.client._SLOW_RESPONSE_SECONDS`) logs at WARNING instead, with the threshold, so it shows under the default log configuration. Facet-profile reads run outside this admission, on the process's shared facet profile cache (`materialise.shared_profiles`), which Search and Datasets share when mounted beside it and naming the same roots.
- **Contract schema and refusal typing**: Responses conform to fixed Pydantic models, served as JSON Schema at `GET /api/graph/schema`. Error envelopes use RFC 9457 problem details.
- **Source documentation reader (`materialise.provenance_doc`)**: `GET /api/graph/documentation/{dcid}` submits one whole `read_provenance_doc` call to the DCP-read executor, since its own five reads each depend on the last. It keeps each document in a single-flight cache keyed by the dcid, the request's origin catalogue build and the graph generation, holding up to `GRAPH_PROVENANCE_DOC_CACHE_SIZE` documents. Concurrent requests for one source share one read, a repeat request within a generation and build reads nothing, and a read that fails or that the executor refuses is not kept. Currencies and granularity stay out of the cached document and come from the coverage index on each response. It also fills `scope_and_units.years_covered`, rolled up from the same batched read. `scope_and_units.countries` has no graph answer at all; see the endpoint's own section below.

---

## Endpoints

Every response, success or refusal, carries `x-dfo-trace-id`, `x-dfo-commit`, and a `Server-Timing: total` header. Graph success bodies carry no body-level trace id; the header is the channel.

Reads that depend only on loaded tables or contain permanent gaps answer `Cache-Control: public, max-age=300` with a strong `ETag` and honour `If-None-Match` with `304`. A body with a gap that comes from a failed read or an unloaded index answers `no-store` with no `ETag`, so a shared cache holds no transient fault (`dfo_platform.static_response.gapped_public_json`). `POST` routes carry no cache headers.

### POST /api/graph/resolve

Resolves a batch of 1 to 25 free-text strings to references the graph holds. A result names one entity (`resolved`), the candidates the text names (`ambiguous`), nothing (`not_found`), or an entity the graph does not hold (`not_held`). Countries, continents, UN subregions, continental unions, organisations, DAC aggregates, places below a country, membership groups (for example G7, G20, DAC, the EU, OECD) and, on request, codes are in scope; `geo.region` (loose undata-geo groupings such as WHO member-state lists) is excluded. A `resolved` or `ambiguous` outcome names entities that the input text itself names, by a listed name, alias, or code. A spelling-corrected match, for example "Lagos" resolving to Laos, comes back as `not_found` instead, with the corrected name as the first near match.

A text that resolves to nothing on the first pass gets one retry, with a leading "the" stripped (case-insensitive, only when followed by whitespace) and any trailing run of `.`, `,`, `;`, `:`, `!`, or `?` characters stripped. The retry's result replaces the miss only when it produces a survivor. "the EU" resolves to the European Union this way, and "Kenya." resolves to `country/KEN`. A name that carries the article, such as "the Gambia", resolves on the first pass and is untouched by the retry.

A text with no listed-name survivor takes the description-arc fallback ladder, which admits a node only when the text is one of that node's own listed names.

#### Request constraints

- `texts`: array of 1 to 25 strings. Each is stripped of surrounding whitespace and must then be 1 to 200 characters.
- `kinds`: the kinds a candidate may have, one or more of `country`, `continent`, `subregion`, `continental_union`, `place_other`, `aggregate`, `organisation`, `group`, `code` and `other` (`resolve_kinds` on `GET /api/graph/capabilities`). Omitted, it admits the first eight, so a request without `kinds` equals one naming them. A candidate of another kind is removed, and a text left with none returns `not_found` with the removed candidates as `near_matches`.
- `code_system`: the graph id of one code system (`CRSPurposeCodeEnum`) that narrows the `code` kind. Naming it without `code` in `kinds` is refused `400 request_invalid` at `/code_system`.
- `expand`: `["members"]` puts a group's or region's members inline in place of the link. At most 256 members are inline. The default is `[]`.
- Extra fields are forbidden (`extra="forbid"`).

#### Result grammar

All outcomes return HTTP 200 with one result per input text, index-aligned with `texts`. Every result has the same fields, and a field the status does not use is `null`.

| Status      | Condition                                                                                                                                | Fields set                                                                                      |
| :---------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| `resolved`  | One entity the input text itself names.                                                                                                  | `entity`, `matched`, `reading`; for a group or a region also `projection`, `member_count`, `members` |
| `ambiguous` | Two or more entities the input text itself names, in the resolver's order.                                                               | `candidates`                                                                                    |
| `not_found` | No candidate of the requested kinds survived. A spelling-corrected match lands here too.                                                 | `near_matches`                                                                                  |
| `not_held`  | The resolver named a real entity, but this deployment's graph holds no node for it.                                                      | `entity`, with `kind` `null`                                                                    |

Each `entity`, `candidates` entry and `near_matches` entry is `{ref, label, kind}`. `ref` is the graph dcid and `kind` is one of the 16 kinds on `GET /api/graph/capabilities`. `matched` is `{text, as}`: the listed text the input matched, and whether it is the entity's `name`, an `alias`, or a `code`. `reading` is `members` when the text names a group's members ("EU countries") and `entity` otherwise.

A group or a region resolves like any entity. A group resolves to `kind` `group` whether or not the graph holds a node for it, so "G7" resolves to `GroupOfSeven` with no graph read. A region is a `continent` or a `subregion` that the membership policy holds, and a region the policy does not hold has `null` for `projection`, `member_count` and `members`. For both, the entity's `label` is the membership authority's canonical name, as the node route gives it, `projection` is `countries` (the default projection: countries that belong to the United Nations today), `member_count` is the policy's count, and `members` is the link to `GET /api/graph/node/{dcid}/members` for the relation (`members` for a group, `countries` for a region). With `expand: ["members"]`, `members` is a list of `{ref, label}` instead. A group's members come from the resolver's registry and carry resolvekit's names, and a region's members take the membership policy's canonical names, as the members route labels them. The largest group is the United Nations at 193 members; "G20" resolves to its 19 country members, and the EU and the African Union are not added to it.

`not_found.near_matches` lists the candidates a `kinds` filter removed, then up to three rejected hints, one per `ref`. They are hints for a person; a program should pick none by id. `not_held.entity.ref` does not open on `GET /api/graph/node/{dcid}`, which answers `404 node_not_held` for it. The move is the same as for `not_found`: report the gap and do not retry.

`status` is a closed union of the four values above. A client that mirrors this contract by hand should treat an unrecognised `status` as a surfaced failure and not fold it into `not_found` or `not_held`.

#### Resolution pipeline

1. **Component resolve**: `entity_resolution.EntityResolver.resolve(texts, kinds=..., code_system=...)` resolves the whole batch in one call: resolvekit matching, the listed-name gate, the description-arc fallback ladder, the code index for the `code` kind, and graph confirmation.
2. **Membership**: A group or region result takes its count, link, members and member names from the membership policy, with no graph read.
3. **Result assembly**: Each resolution maps to a result, index-aligned with the input batch.

One absolute deadline of `GRAPH_RESOLVE_TIMEOUT_SECONDS` covers the resolvekit job and every DCP read. When it passes the request returns `504 timeout`.

#### Request example

```json
{
  "texts": ["Kenya", "Congo", "G7", "North America", "Atlantis Prime"]
}
```

#### Response example

The example omits the `null` members each result carries. The response comes from the app built with the `tests/graph_api` fixtures (`make_client` and the graph nodes of `test_accept_resolve.py`).

```json
{
  "results": [
    {
      "input": "Kenya",
      "status": "resolved",
      "entity": {"ref": "country/KEN", "label": "Kenya", "kind": "country"},
      "matched": {"text": "Kenya", "as": "name"},
      "reading": "entity"
    },
    {
      "input": "Congo",
      "status": "ambiguous",
      "candidates": [
        {"ref": "country/COD", "label": "Congo [DRC]", "kind": "country"},
        {"ref": "country/COG", "label": "Congo [Republic]", "kind": "country"}
      ]
    },
    {
      "input": "G7",
      "status": "resolved",
      "entity": {"ref": "GroupOfSeven", "label": "Group of Seven", "kind": "group"},
      "matched": {"text": "G7", "as": "alias"},
      "reading": "entity",
      "projection": "countries",
      "member_count": 7,
      "members": "/api/graph/node/GroupOfSeven/members?relation=members&projection=countries"
    },
    {
      "input": "North America",
      "status": "resolved",
      "entity": {"ref": "northamerica", "label": "North America", "kind": "continent"},
      "matched": {"text": "North America", "as": "name"},
      "reading": "entity",
      "projection": "countries",
      "member_count": 23,
      "members": "/api/graph/node/northamerica/members?relation=countries&projection=countries"
    },
    {"input": "Atlantis Prime", "status": "not_found", "near_matches": []}
  ]
}
```

With `expand: ["members"]` the G7's `members` is the list of its seven countries:

```json
{
  "results": [
    {
      "input": "G7",
      "status": "resolved",
      "entity": {"ref": "GroupOfSeven", "label": "Group of Seven", "kind": "group"},
      "matched": {"text": "G7", "as": "alias"},
      "reading": "entity",
      "projection": "countries",
      "member_count": 7,
      "members": [
        {"ref": "country/CAN", "label": "Canada"},
        {"ref": "country/DEU", "label": "Germany"},
        {"ref": "country/FRA", "label": "France"},
        {"ref": "country/GBR", "label": "United Kingdom"},
        {"ref": "country/ITA", "label": "Italy"},
        {"ref": "country/JPN", "label": "Japan"},
        {"ref": "country/USA", "label": "United States"}
      ]
    }
  ]
}
```

---

### GET /api/graph/node/{dcid}

Describes one node under one envelope. Multi-segment DCIDs containing slashes (such as `country/KEN` or `dc/base/UNSD_M49_SubRegion`) are supported via a custom URL path convertor matching `[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*`.

The `dcid` path parameter is capped at 512 characters, published as `caps.max_dcid_chars` on GET /api/graph/capabilities, which is also the cap `GET /api/graph/documentation/{dcid}` enforces. A longer value is refused `400 request_invalid` before any DCP read runs.

`as_of` (an ISO date) dates a group's or region's membership. On a membership that records no dates it is refused `400 request_invalid` at `/as_of`, and any other kind ignores it.

#### Envelope

Every node carries the same members, plus one detail block named for its `kind`. `other` has no block.

| Member        | Content                                                                                                                                  |
| :------------ | :--------------------------------------------------------------------------------------------------------------------------------------- |
| `ref`         | The node's dcid.                                                                                                                         |
| `label`       | The node's name. For a Provenance, its parent Dataset's name where it has one. For a group, and a region the membership policy holds, the membership authority's canonical name, as resolve gives it. `null`, and named in `gaps`, when the graph gives none. |
| `kind`        | One of the 16 kinds below.                                                                                                               |
| `types`       | The node's `typeOf` values, as the graph holds them.                                                                                     |
| `description` | The node's prose in the graph; `null` when it has none.                                                                                  |
| `links`       | Each list relation the node has, mapped to its route on `GET /api/graph/node/{dcid}/members`. `{}` when none.                           |
| `gaps`        | The member path of every value this door cannot know. A value it cannot know is `null`; a verified empty list is `[]`.                  |
| `built_from`  | The build of each table the response read: `code_index`, `origin_catalogue`, `coverage_index` or `membership`.                           |

Every reference is `{ref, label}`, with the dcid in `ref`. A label skips a name equal to its own dcid, a junk triple some imports write.

#### Kinds and blocks

The kind comes from `entity_resolution.kinds.classify`, which resolve calls too, over the same code index and membership policy, so a node has one kind on both routes. It reads, in order, and the first that holds wins: the membership policy (a group it holds is `group`, whatever the graph says of the id), `typeOf` (`StatisticalVariable`, `Provenance`, `Source`, `Dataset`), the code index (a code system's graph id, a code row's dcid), the membership policy again (a region it holds is `continent` when its authority, resolvekit's UN M49 store, classes it a continent, else `subregion`, whatever its graph types: `MembershipPolicy.kind_of`), unit types, the code index's parties (`total` or `residual` parties are `aggregate`), and the resolver's place and organisation types. Anything else is `other`.

| `kind`              | Block (same name)                                                  | `links`                                                           |
| :------------------ | :----------------------------------------------------------------- | :---------------------------------------------------------------- |
| `stat_var`          | `definition`, `availability`                                       | `facets`                                                          |
| `code_system`       | `property`, `plain_name`, `declarations`, `code_count`             | `codes`                                                           |
| `code`              | `code_system`, `property`, `code`, `parent`, `synonyms`            | `children`, when the code has children                            |
| `group`             | `authority`, `as_of`, `effective_date`, `projections`              | `members`                                                         |
| `country`, `continent`, `subregion`, `continental_union`, `place_other` | `contained_in`, `membership` (set for a region the policy holds) | `countries`, for a region the policy holds |
| `organisation`, `aggregate` | `roles`                                                    | none                                                              |
| `unit`              | `unit_class`, `currency`, `currency_noun`, `price_basis`, `changes_magnitude` | none                                                   |
| `source`            | the source block                                                   | `datasets`                                                        |
| `dataset`, `provenance` | the source block                                               | `variables`                                                       |
| `other`             | none                                                               | none                                                              |

The source block holds `publisher` (the `source` arc's target), `parent` (the `isPartOf` arc's target), `license`, `license_type`, `url`, `source_data_url`, `description_url`, `transformation_logic_url`, `last_data_refresh_date`, `next_data_refresh_date`, `earliest_observation_date`, `latest_observation_date`, `freshness` (`release_frequency`, `release_frequency_label`, `refresh_type`), `data_commons` and `documentation` (the path of the node's source document). `data_commons` is `{id, label}` for the Data Commons that serves an admitted provenance, from the request's origin catalogue with no graph read, and `null` for a provenance the catalogue does not admit, a Dataset or a Source. `id` is `dfo`, `base` or `un_system`. A dcid under `undata/` is read from the UN System Data Commons.

A membership block (a group's `group`, a region's `membership`) holds `authority` (`name`, `version`), `as_of` (the date asked for, `null` for today), `effective_date` (the latest membership change on or before that date, `null` for an undated membership) and `projections`: one `{projection, count, complete}` entry for each of `countries`, `other_states`, `territories`, `former_states` and `institutions`. `count` is `null` and the projection is named in `gaps` when the authority does not record it, which holds for `institutions` everywhere. `complete` is `false` when the authority does not record the projection, and when it holds the projection only in part (`membership_incomplete`): for a region that covers `territories` and `former_states`, because resolvekit holds 18 territories as no country and places almost no former state in a region.

An organisation's or aggregate's `roles` are the role properties a node of its types can fill, from the role ranges the graph declares. When that read fails `roles` is `null` with a transient gap.

#### Variable card

A `stat_var` block holds the variable's `definition` and its `availability`.

`definition` (`graph_api.card_definition`) names the `population_type`, `measured_property`, `stat_type` and `measure`, the `roles` the variable's observations carry (`observationAbout` when it declares none), the publisher's `source_indicator_codes`, its `constraints` as `{property, value}` pairs ordered by property dcid, and `definitions`, the graph's prose for each part in order: the variable's own, the measure's, then each constraint value's. A World Development Indicators variable's own prose (`worldBank/`) leaves out the series code its import appends, "(WorldBank Series Code: ...)", and one whose description is that code alone has none. Each part stays its own entry. A variable the code index holds takes its shape and constraints from its row; any other takes them from its graph arcs. No graph node holds a measure's definition yet, so `measure` carries the gap `stat_var.definition.measure.definition`.

`availability` (`graph_api.availability`) says where DFO holds observations of the variable, under `scope` `dfo`:

- `summary`: the places covered by kind (`places`), or for a role-bearing variable the entities per role (`entities`), with `first_year`, `last_year`, `sources` and `facets` counts.
- `default`: the facets a data request naming no place, years or facet receives (`materialise.facets.served_facets`), with their source, unit, measurement method, period, span, `places` and `without_data`. `null` when no facet dispatches such a request.
- `sources`: up to ten sources, each with its `source` (`{ref, label, documentation}`) and `series`. A series groups a source's facets by measurement method, period and scaling, and carries its `units` fold (`count`, the two `leading` units by places covered, `more`, and `national_currency` when the units are each covered place's own currency) and a `binding` of the `scope.facets` values that request the series from Datasets. `more_sources` counts the rest.

A role-bearing DFO variable reads the coverage index's facet summary; every other variable reads its facet profile. A failed profile read serves `availability: null` with the transient gap `stat_var.availability`, and the response answers `no-store`. A surface with no profile cache serves `availability: null` with the permanent gap `stat_var.availability.not_served`, and the response answers `public`. Spans are unions over places and claim no continuity at any one place.

The request waits for the names of units and methods, the kind of each place and the national currencies until just before its deadline. These reads outlive the request and fill their caches, so a later request finds them. Without them, whether a read failed, ran past the deadline or found every DCP-read worker busy, a card leaves those labels `null`, `summary.places` null and no unit folded into `national_currency`, each with a transient gap.

#### Response example (`group`)

The response comes from the app built with the `tests/graph_api` fixtures (`make_client` over a fake graph that holds no node for the group, since a group opens with or without a graph node).

```json
{
  "ref": "GroupOfSeven",
  "label": "Group of Seven",
  "types": [],
  "description": null,
  "links": {"members": "/api/graph/node/GroupOfSeven/members?relation=members&projection=countries"},
  "gaps": ["group.projections.institutions"],
  "built_from": {
    "code_index": "1be06f70b06f447fbc48644d99f3d023f30bf8987b2ccdf0b82c6ca73228e7e7",
    "membership": "resolvekit 0.1.13, data 2026.10"
  },
  "kind": "group",
  "group": {
    "authority": {"name": "resolvekit curated membership", "version": "resolvekit 0.1.13, data 2026.10"},
    "as_of": null,
    "effective_date": "1976-01-01",
    "projections": [
      {"projection": "countries", "count": 7, "complete": true},
      {"projection": "other_states", "count": 0, "complete": true},
      {"projection": "territories", "count": 0, "complete": true},
      {"projection": "former_states", "count": 0, "complete": true},
      {"projection": "institutions", "count": null, "complete": false}
    ]
  }
}
```

#### Response example (`continent`)

```json
{
  "ref": "northamerica",
  "label": "North America",
  "types": ["Continent"],
  "description": null,
  "links": {"countries": "/api/graph/node/northamerica/members?relation=countries&projection=countries"},
  "gaps": ["continent.membership.effective_date", "continent.membership.projections.institutions"],
  "built_from": {
    "code_index": "1be06f70b06f447fbc48644d99f3d023f30bf8987b2ccdf0b82c6ca73228e7e7",
    "membership": "resolvekit 0.1.13, data 2026.10"
  },
  "kind": "continent",
  "continent": {
    "contained_in": [],
    "membership": {
      "authority": {"name": "resolvekit UN M49 containment", "version": "resolvekit 0.1.13, data 2026.10"},
      "as_of": null,
      "effective_date": null,
      "projections": [
        {"projection": "countries", "count": 23, "complete": true},
        {"projection": "other_states", "count": 0, "complete": true},
        {"projection": "territories", "count": 15, "complete": false},
        {"projection": "former_states", "count": 1, "complete": false},
        {"projection": "institutions", "count": null, "complete": false}
      ]
    }
  }
}
```

#### Response example (`stat_var`)

The response comes from the app built with the variable-card fixtures of `tests/graph_api/test_node_card.py`, with a population type, a measured property and a statistical type added to the variable's node. The example keeps the first of the source's two series.

```json
{
  "ref": "GDP_Nominal",
  "label": "GDP, current prices",
  "types": ["StatisticalVariable"],
  "description": "Gross domestic product at current prices.",
  "links": {"facets": "/api/graph/node/GDP_Nominal/members?relation=facets"},
  "gaps": ["stat_var.definition.stat_type.label", "stat_var.definition.measure"],
  "built_from": {
    "code_index": "1be06f70b06f447fbc48644d99f3d023f30bf8987b2ccdf0b82c6ca73228e7e7",
    "origin_catalogue": "restrictive-catalogue-6fde3fb68115",
    "coverage_index": "test-coverage-stamp-build"
  },
  "kind": "stat_var",
  "stat_var": {
    "definition": {
      "population_type": {"ref": "EconomicActivity", "label": "Economic activity"},
      "measured_property": {"ref": "gdp", "label": "gross domestic product"},
      "stat_type": {"ref": "measuredValue", "label": null},
      "measure": null,
      "roles": ["observationAbout"],
      "source_indicator_codes": [],
      "constraints": [],
      "definitions": [
        {
          "of": {"ref": "GDP_Nominal", "label": "GDP, current prices"},
          "text": "Gross domestic product at current prices."
        }
      ]
    },
    "availability": {
      "scope": "dfo",
      "summary": {
        "places": {"country": 2},
        "entities": null,
        "first_year": "2000",
        "last_year": "2020",
        "sources": 1,
        "facets": 2
      },
      "default": {
        "facets": ["fUSD"],
        "source": {
          "ref": "provenance/IMFWorldEconomicOutlook",
          "label": "provenance/IMFWorldEconomicOutlook",
          "documentation": "/api/graph/documentation/provenance/IMFWorldEconomicOutlook"
        },
        "unit": {"ref": "CurrencyUSD", "label": "US dollars"},
        "measurement_method": null,
        "observation_period": "P1Y",
        "first_year": "2000",
        "last_year": "2020",
        "places": 2,
        "without_data": 0
      },
      "sources": [
        {
          "source": {
            "ref": "provenance/IMFWorldEconomicOutlook",
            "label": "provenance/IMFWorldEconomicOutlook",
            "documentation": "/api/graph/documentation/provenance/IMFWorldEconomicOutlook"
          },
          "places": {"country": 2},
          "entities": null,
          "first_year": "2000",
          "last_year": "2020",
          "facets": 2,
          "series": [
            {
              "measurement_method": null,
              "observation_period": "P1Y",
              "scaling": null,
              "first_year": "2000",
              "last_year": "2020",
              "places": 2,
              "observations": 42,
              "units": {
                "count": 1,
                "leading": [{"ref": "CurrencyUSD", "label": "US dollars", "places": 2}],
                "more": 0,
                "national_currency": null
              },
              "binding": {
                "provenance": "provenance/IMFWorldEconomicOutlook",
                "measurementMethod": null,
                "unit": "CurrencyUSD"
              }
            }
          ]
        }
      ],
      "more_sources": 0
    }
  }
}
```

#### Property allowlist

Reads retrieve a fixed property allowlist to avoid oversized properties such as polygon geometries (`graph_api.doors.node._ALLOWLIST`):

```python
(
    "typeOf", "name", "description", "definition", "populationType",
    "measuredProperty", "observationProperties", "constraintProperties",
    "sourceIndicatorCode", "license", "licenseType", "source",
    "sourceReleaseFrequency", "automaticRefreshType", "url", "sourceDataUrl",
    "descriptionUrl", "dataTransformationLogic", "lastDataRefreshDate",
    "nextDataRefreshDate", "earliestObservationDate", "latestObservationDate",
    "isPartOf", "provenanceCategory", "containedInPlace", "iso4217AlphaCode"
)
```

A missing node answers `404 node_not_held`. A group the membership policy holds opens with or without a graph node.

---

### POST /api/graph/nodes

Reads up to 50 nodes in one request, as at most one executor job. A client that needs several nodes at once sends them here: parallel `GET /api/graph/node/{dcid}` calls each take a worker of the shared pool (`GRAPH_NODE_POOL_SIZE`, 4 by default).

The door makes one allowlisted arc read for every distinct dcid, classifies each held node, then makes one batched name read for the label targets, skipped when none has a target. Every `stat_var` in the request takes one facet profile read. Each entry is the body `GET /api/graph/node/{dcid}` returns for that dcid. A dcid the graph does not hold gets `{ref, status: "not_held"}` where the GET route answers `404 node_not_held`.

The door keeps each entry it builds with no transient gap (`graph_api.node_cache`), up to `GRAPH_NODE_ENTRY_CACHE_SIZE` entries, keyed by the graph generation, the code index, origin catalogue, coverage and membership builds, and the dcid. An entry is stored under the builds its `built_from` names, and expires with its generation's cache lifetime. A batch reads only the dcids it misses there, so a repeat of a batch within one generation and set of builds makes no graph read. A GET without `as_of` reads and fills the same entries; a GET with `as_of` always reads. A generation change clears them.

#### Request constraints

- `dcids`: array of 1 to 50 dcids, published as `caps.max_node_batch_dcids` on GET /api/graph/capabilities. Each matches the GET route's path syntax (`[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*`) and its 512-character ceiling. A list outside these bounds, a malformed dcid, or an extra key is refused `400 request_invalid` before any DCP read runs.

```json
{ "dcids": ["country/KEN", "country/XYZ"] }
```

#### Response

`nodes` is index-aligned with `dcids`, and a repeated dcid repeats its entry. The response comes from the app built with `make_client` and a fake graph holding `country/KEN` and `africa`.

```json
{
  "nodes": [
    {
      "ref": "country/KEN",
      "label": "Kenya",
      "types": ["Country"],
      "description": null,
      "links": {},
      "gaps": [],
      "built_from": {"code_index": "1be06f70b06f447fbc48644d99f3d023f30bf8987b2ccdf0b82c6ca73228e7e7"},
      "kind": "country",
      "country": {
        "contained_in": [{"ref": "africa", "label": "Africa"}],
        "membership": null
      }
    },
    {"ref": "country/XYZ", "status": "not_held"}
  ]
}
```

---

### GET /api/graph/node/{dcid}/members

Lists what belongs to a node under one `relation`. The route must be mounted ahead of `GET /api/graph/node/{dcid}`, whose dcid convertor accepts slashes and would otherwise take `<dcid>/members` as one dcid. A node's `links` and a resolve result's `members` give the route with its `relation`.

| `relation`  | Node                                 | Items                                                                      | Owner                                              | Order    |
| :---------- | :----------------------------------- | :------------------------------------------------------------------------- | :------------------------------------------------- | :------- |
| `codes`     | a code system's graph id             | the system's codes, `{ref, label, code, parent}`                           | code index                                         | code     |
| `children`  | a code                               | the code's direct children, same items; a leaf answers an empty list       | code index                                         | code     |
| `datasets`  | a source                             | the source's upstream Datasets; a provenance with no Dataset is its own entry | origin catalogue, through the source index      | dcid     |
| `variables` | a Dataset or a Provenance            | the variables DFO indexes under it                                         | code index provenance map and origin catalogue     | dcid     |
| `members`   | a group                              | the group's members under a projection and an `as_of` date                 | membership policy                                  | dcid     |
| `countries` | a region                             | the region's countries under a projection                                  | membership policy                                  | dcid     |
| `facets`    | a stat var                           | the variable's facets, each as the card's series reads them                | facet profile and coverage index                   | facet id |

#### Parameters

- `relation`: required.
- `limit`: items per page, 1 to 1000 (`MEMBERS_PAGE_MAX`); default 200 (`MEMBERS_PAGE_DEFAULT`).
- `cursor`: the `next_cursor` of the previous page. The cursor is opaque, bound to the node, relation, projection, `as_of` date, order and, for `facets`, the filter values. It continues after the last item's sort key, with the dcid breaking ties, so a page read after a rebuild repeats no item and skips none present in both builds. A cursor that does not decode or belongs to another walk is refused at `/cursor`.
- `projection`: for `members` and `countries` only, one of `countries` (the default), `other_states`, `territories`, `former_states` or `institutions`. Any other relation refuses it at `/projection`.
- `as_of`: for `members` and `countries` only, an ISO date. An undated membership (every region, and the 20 groups whose authority records no dates) refuses it at `/as_of`.
- `source`, `unit`, `measurement_method`: for `facets` only, filter the facets by provenance, unit and measurement method dcid. An empty `measurement_method` selects the facets that state no method. Any other relation refuses a filter at its name.

#### Page

`ref`, `relation`, `projection` and `as_of` echo the walk. `items` holds the page. `next_cursor` is `null` when the page ends the list. `complete` is true when the walk reached the end of a list the owner holds in full, and `total` is the list's length, `null` where the list is partial. `gaps` holds the code of each way the list is partial and `built_from` names the build of each table the list read.

Items carry the owner's label: the code index's for codes, the catalogue's for Datasets, and the membership authority's canonical name for members. A `variables` page takes its labels from one batched graph name read on the DCP-read executor under `GRAPH_NODE_TIMEOUT_SECONDS`. A label is `null` where its owner holds no name. A `facets` item is `{ref, label, source, unit, measurement_method, observation_period, scaling, first_year, last_year, observations, places}`, where `ref` is the facet id, a hash of the facet's metadata.

Gaps on a list:

- `membership_incomplete`: the authority holds the projection only in part. The page is `complete: false`.
- `projection_not_recorded`: the authority records no members for the projection (`institutions`). The page has no items.
- `variables_not_indexed`: a Provenance of the Dataset has no entry in the code index.
- `labels_not_read`: the label read failed, ran past the deadline or found every DCP-read worker busy, and the labels are `null`.
- `facets_not_read`: the variable's facet profile could not be read. The page lists nothing.
- `facets_not_served`: the surface reads no facet profiles, because it has no profile cache. The page lists nothing. The gap is permanent.
- `stat_var.availability.countries_only`: the variable's default place scope fell back to the countries with data, as on its card, so each facet's `places` counts countries only. The gap is permanent.

A page with `labels_not_read` or `facets_not_read` answers `no-store`.

#### Refusals

- `400 request_invalid` with a pointer: `/cursor`, `/relation` (the node is held and has no such list), `/projection`, `/as_of`, `/limit`, a filter name, or `/dcid` for a dcid over 512 characters.
- `404 node_not_held` at `/dcid` for a dcid the deployment does not hold, as the node route answers it. A group or region the membership policy holds is held with or without a graph node, so it answers `400` at `/relation` for a relation it lacks. The hold is read only when a list is refused.
- `503 service_unavailable` when the surface has published no membership policy, or when a `facets` request's node types were not read by the deadline.

#### Response example (`codes`)

The response comes from the app built with `make_client` and no graph nodes, since the code index is in process.

```json
{
  "ref": "CRSPurposeCodeEnum",
  "relation": "codes",
  "projection": null,
  "as_of": null,
  "items": [
    {
      "ref": "CRSPurposeCode100",
      "label": "social infrastructure and services",
      "code": "100",
      "parent": null
    },
    {
      "ref": "CRSPurposeCode110",
      "label": "education",
      "code": "110",
      "parent": "CRSPurposeCode100"
    }
  ],
  "next_cursor": "eyJiIjoiZTI0MTM0NGY5MGQyMjM0NCIsImsiOiIxMTAiLCJkIjoiQ1JTUHVycG9zZUNvZGUxMTAifQ",
  "complete": false,
  "total": 289,
  "gaps": [],
  "built_from": {"code_index": "1be06f70b06f447fbc48644d99f3d023f30bf8987b2ccdf0b82c6ca73228e7e7"}
}
```

#### Response example (`countries`, a partial projection)

```json
{
  "ref": "northamerica",
  "relation": "countries",
  "projection": "territories",
  "as_of": null,
  "items": [
    {"ref": "country/ABW", "label": "Aruba"},
    {"ref": "country/AIA", "label": "Anguilla"},
    {"ref": "country/BES", "label": "Bonaire, Sint Eustatius and Saba"}
  ],
  "next_cursor": "eyJiIjoiZTQzYWQ1ZjY1MDI1YmE5NyIsImsiOiJjb3VudHJ5L0JFUyIsImQiOiJjb3VudHJ5L0JFUyJ9",
  "complete": false,
  "total": null,
  "gaps": ["membership_incomplete"],
  "built_from": {"membership": "resolvekit 0.1.13, data 2026.10"}
}
```

#### Response example (refusal)

```json
{
  "type": "/codes/request_invalid",
  "title": "Bad Request",
  "status": 400,
  "detail": "'CRSPurposeCodeEnum' has no children list",
  "code": "request_invalid",
  "pointer": "/relation",
  "trace_id": "0250cb03b740477893161f485abc825c",
  "capability_version": "graph_<token>",
  "retryable": false
}
```

---

### GET /api/graph/roots

Names where to start listing: each code system with its code count (`code_systems`), each group (`groups`) and region (`regions`) with its member count under the default projection, and each source with its Dataset count (`sources`). Each entry is `{ref, label, count}`, each list is ordered by dcid, and `built_from` names the builds of the code index, origin catalogue and membership. A refusal `503 service_unavailable` follows an unpublished membership policy.

The example keeps the first entry of each list. The response comes from the app built with `make_client` and the restrictive test catalogue.

```json
{
  "code_systems": [{"ref": "CRSCooperationModalityEnum", "label": "modality", "count": 35}],
  "groups": [{"ref": "EuropeanUnion", "label": "European Union", "count": 27}],
  "regions": [{"ref": "AustraliaAndNewZealand", "label": "Australia and New Zealand", "count": 2}],
  "sources": [
    {
      "ref": "dc/s/BankForInternationalSettlementsBis",
      "label": "Bank for International Settlements (BIS)",
      "count": 0
    }
  ],
  "built_from": {
    "code_index": "1be06f70b06f447fbc48644d99f3d023f30bf8987b2ccdf0b82c6ca73228e7e7",
    "origin_catalogue": "restrictive-catalogue-6fde3fb68115",
    "membership": "resolvekit 0.1.13, data 2026.10"
  }
}
```

---

### GET /api/graph/documentation/{dcid}

Assembles one source's documentation document, in the field names below. `dcid` accepts either a `Dataset` dcid (the catalogue's own grain, see below) or a `Provenance` dcid: a Provenance with a parent Dataset climbs to it and rolls up across every one of that Dataset's own children, not just the given node alone; a Provenance with no parent (an orphan, e.g. the ONE-minted `provenance/OECDCreditorReportingSystem`) is treated as its own single child. A `Source` dcid (the publishing organisation, e.g. `dc/s/WorldBank`) is refused `400 request_invalid`: this door documents a publication, not a publisher. A UN System Data Commons agency provenance (`undata/p/SDG`) is read from the UN deployment and documented the same way.

The response answers `public, max-age=300` with an `ETag` over its body, or `no-store` when it asked for coverage facts while the coverage index was unreadable. An unreadable index can answer on a later read.

The graph's own three-level chain is **Source** (the publishing organisation) -> **Dataset** (the publication, e.g. "World Development Indicators") -> **Provenance** (one import). Several Provenance nodes often share one Dataset (two ACS subject tables, two GeoBoundaries country imports), so the catalogue this door serves lists Dataset nodes, and the documentation document is assembled by rolling up the Provenance nodes beneath one.

#### Roll-up rules

- A Dataset's children are the `Provenance` nodes that name it in `isPartOf` and that the request's origin catalogue admits. A withheld or excluded child counts toward no member. A Dataset with no admitted child answers `404 node_not_held`, since the source index lists no entry for it. A Provenance's parent Dataset is its first `isPartOf` parent typed `Dataset`. A Provenance whose parents are all something else (a Source, a `ProvisionalNode`, a node the graph does not hold) is an orphan.
- `details.description`, `details.source_url`, `license.type`, `license.update_cadence` and `scope_and_units.years_covered` come from `origin_catalogue.card_facts.card_facts_of`, the function the catalogue job computes the source index's card facts with (see `GET /api/graph/sources`).
- `details.name_of_source`, `details.source_url`, `details.publisher` read from the Dataset node directly, no roll-up. `source_url` falls back to the sole child's own `url` when the Dataset has exactly one child and none of its own. `name_of_source` falls back to the given node's own `name` for an orphan, or when the resolved Dataset carries none; both reads apply the same self-referential-name guard `publisher` uses on the plain node door (above). A `name` value equal to its own dcid is never served, falling through to the next name in the chain instead.
- `details.source` and `details.data_commons` come from the request's origin catalogue. `source` is the provenance table's source (`{ ref, label }`, the registry's name for the `Source` node; for a UN agency, the agency provenance itself) and `data_commons` its Data Commons (`{ id, label }`). A Provenance dcid uses its own catalogue row. A Dataset dcid uses its admitted children's rows and serves each when they share one value; otherwise it is `null` and gapped, so a Dataset whose children sit under two sources serves `details.source` `null`.
- `license.last_update` is the max across every value every child asserts for `lastDataRefreshDate`, flattened across children; `license.next_expected_update` is the min across `nextDataRefreshDate`, same flattening. A child that asserts the same property twice contributes both values, not just whichever the graph returned first.
- `license.type` and `license.update_cadence` serve the value only when every child asserts exactly one value for the property and all children's values agree. A child that carries no value at all empties the field the same as an outright cross-child conflict does. One declaring child does not speak for a publication the rest are silent on, and the path joins `gaps`. `license.type` is the label of the agreed `licenseType` node, its `name` in the graph (e.g. `Creative Commons Attribution License` for `CreativeCommonsAttributionLicense`), or its dcid when the graph holds no name for it (`origin_catalogue.card_facts.licence_labels_of`). One further node read fetches the name. `license.update_cadence` converts `sourceReleaseFrequency`'s own ISO 8601 duration token the same way `freshness.release_frequency_label` already does on the plain node door (`materialise.cadence`).
- `details.citation` (`licenseAttribution`), `details.source_data_url` (`sourceDataUrl`, the file the pipeline downloads) and `license.url` (`license`, the licence text on the publisher's site) follow the same every-child-agrees rule as `license.type` and serve the stored value unchanged. Datasets carry no `sourceDataUrl` or `license` of their own, so both come from the children. `details.source_url` is the publisher's landing page.
- `details.description` is `null` on any Dataset with more than one child. No honest roll-up exists. A single-child Dataset (including an orphan's own degenerate one-child case) serves that one child's `description`.
- `scope_and_units.years_covered` is the earliest `earliestObservationDate` across every child that carries one, joined to the latest `latestObservationDate` across every child that carries one, as `"{start}-{end}"` (e.g. `"1973-2024"`). `null` when no child carries either property, present on about 6.7% of provenances in the dev graph's own random sample. `scope_and_units.countries` is permanently `[]`: no graph structure answers "which countries does this source cover" for a general source. See `materialise.provenance_doc`'s own docstring, "On scope_and_units.countries", for the evidence and what would unblock it.
- `scope_and_units.currencies` lists, sorted, the ISO 4217 code of every currency (e.g. `["EUR", "USD"]`) that a coverage series' facet carries as its `Currency` unit, across all of the document's Provenances (the given orphan, or every child of the Dataset). It reads `series_coverage.CoverageIndex.provenance_currencies`. Only facets the coverage rows use count, the same series `granularity` reads, so facet metadata no series uses adds no currency. A Provenance with no currency-valued series or no coverage rows adds none. When no Provenance carries a currency, the list is `[]` and the path `scope_and_units.currencies` is gapped.
- `scope_and_units.granularity` is computed from the same coverage index (`series_coverage.CoverageIndex.provenance_shapes`), e.g. `"Annual, by provider and recipient"`. It is served when every one of the document's Provenances has coverage series and together they share exactly one observation period that `materialise.cadence` labels (`P1Y` reads `Annual`) and the union of their variables' role axes is the flow pair, origin and destination, named by `materialise.roles.axis_noun` in that order. The union is the finest granularity, so a Provenance with one-role totals beside by-creditor variables reads as a two-axis source. A Provenance without coverage rows, two periods, a series without a period, an unlabelled period token, a union of one axis, or a role outside the two flow axes serves `null`.
- Both coverage facts see only role-bearing series. The coverage job keeps series whose variable declares one or two `observationProperties` roles: a two-role series with its donor and recipient, and a one-role series with its entity as recipient and no donor. It drops series keyed by `observationAbout` alone. A Provenance without coverage rows adds no currency and makes `granularity` `null`. A Provenance that holds both kinds is described by its role-bearing series only, so a currency or period that appears only in its `observationAbout` series is not seen.
- Both facts come from the coverage view of the release bound to the request, so a release swap never mixes two releases in one document. A failure reading that snapshot logs one WARNING naming the Provenance dcids and serves both facts empty. A failure deriving one fact logs a WARNING naming the Provenance dcids and serves only that fact empty. The request succeeds in every case.

#### Documents

`transformation_log`, `version_history`, `methodology.description` and `data_notes` come from four Markdown documents on the sole child Provenance: `dataTransformationDescription`, `dataTransformationChangelog`, `sourceMethodologyDescription` and `dataNotes`. The pipeline stores each document MCF-escaped; this door decodes it and parses it against its template with `dfo_provenance_docs`, whose README states the templates. A Dataset with several children serves all four empty, with no error. Text members carry CommonMark, with a paragraph's soft-wrapped lines joined by one space.

- `transformation_log` has one entry per `##` step, with `transformation` the heading and `description` the step's body.
- `version_history` has one entry per `## YYYY-MM-DD <title>` release, newest first. Each has a `date`, a `title`, and `changes`, and each change has a `kind` (`added`, `changed`, `deprecated`, `removed`, `fixed` or `security`, from its `###` group) and a `note`.
- `methodology.description` is the whole methodology document.
- `data_notes` has one entry per bullet of the notes document.

A stored document longer than six characters per byte of `dfo_provenance_docs.MAX_DOCUMENT_BYTES` (the longest escaped form of the largest document the grammar accepts) is refused before decoding, with reason `document_too_large` and `line` `null`. A document that is refused, fails to decode or breaks its template, or a node carrying two distinct values of one document property, leaves its member empty and named in `gaps`, and adds one `document_errors` entry: `member` (the wire path), `line` (1-based in the decoded document, or `null`), and `reason` (a `dfo_provenance_docs.DocumentErrorReason` value, or `conflicting_values`). The rest of the document is still served. `document_errors` is always present, `[]` when every document parsed.

#### `usage`

`usage.api_access` and `usage.python` are `null` for every `dcid` (`graph_api.usage.usage_document`). The document generates no code sample. `GET /api/graph/node/{dcid}/members?relation=variables` lists the variables DFO indexes under a Dataset or a Provenance.

#### Reads

A document costs at most five bounded reads: the node's identity arcs, its parent Dataset when the given node is a Provenance with one, the Dataset's own children via the reverse `isPartOf` arc filtered to `{typeOf:Provenance}`, one batched read covering the publisher's own name and every child's own roll-up properties, and the name of the agreed `licenseType` node when every child names the same one. An orphan Provenance is read in that batch as its own sole child. The batch requests `description`, `url` and the four document properties only when there is exactly one child, and `name` only when there is a publisher, so a many-child Dataset never transfers them. Live-verified against `dc/d/UsCensusBureau_AmericanCommunitySurveyAcs`: 61 Provenance children returned in one page, no `nextToken`. A document already in the route's cache costs no read.

#### `gaps`

`gaps` and `document_errors` sit beside the document's eight members. `gaps` lists every member path that is `null`/`[]` in this particular response, by dotted path, in the order the members are declared (e.g. `"details.citation"`, `"scope_and_units.countries"`). A scalar this door cannot answer serves `null`, and an absent array serves `[]`. `scope_and_units.countries` and both `usage` entries ship `null`/`[]` for every source: countries has no source in this reader, and the `usage` pair for the reason `#usage` above records. These unconditional gaps are published on `GET /api/graph/capabilities` as `source_document_gaps`, so a caller can discover them without fetching a live document. Every other member is named in `gaps` only for the source it came back empty for.

#### Response Example

```json
{
  "details": {
    "name_of_source": "OECD Creditor Reporting System (CRS)",
    "publisher": "Organisation for Economic Co-operation and Development (OECD)",
    "source": {
      "ref": "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
      "label": "Organisation for Economic Co-operation and Development (OECD)"
    },
    "data_commons": { "id": "dfo", "label": "Development Finance Observatory" },
    "description": null,
    "source_url": "https://data-explorer.oecd.org/vis?tm=crs&pg=0&snb=30&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_CRS%40DF_CRS&df[ag]=OECD.DCD.FSD&df[vs]=1.6&isAvailabilityDisabled=false&page=0",
    "source_data_url": "https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",
    "citation": "OECD (2026), Creditor Reporting System (CRS), OECD Data Explorer, https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet (accessed on 2026-09-11)"
  },
  "license": {
    "type": "OECD Terms and Conditions",
    "url": "https://www.oecd.org/en/about/terms-conditions.html",
    "update_cadence": "Quarterly",
    "last_update": "2026-09-11",
    "next_expected_update": null
  },
  "scope_and_units": {
    "countries": [],
    "years_covered": "1973-2024",
    "granularity": null,
    "currencies": []
  },
  "methodology": {
    "description": null
  },
  "data_notes": [],
  "usage": {
    "api_access": null,
    "python": null
  },
  "transformation_log": [
    {
      "transformation": "Download the source data",
      "description": "The complete CRS file is downloaded from the OECD each time the pipeline runs, and replaces the previous copy, so the data reflects the OECD's latest release."
    },
    {
      "transformation": "Exclude donor core contributions to multilaterals",
      "description": "The OECD instructs that donor core contributions to multilateral organisations, reported as ordinary CRS rows flagged `bi_multi = 2`, are excluded to compute bilateral flows. These contributions are already reported in Members' Total Use of the Multilateral System, so keeping them would double count."
    },
    {
      "transformation": "Standardise providers and recipients",
      "description": "Provider and recipient codes are standardised to Data Commons IDs used across the platform, so data can be compared across datasets."
    },
    {
      "transformation": "Select dimensions",
      "description": "Every figure is reported against the year, the provider and the recipient, and four dimensions from the CRS: type of flow, co-operation modality, type of finance and purpose code. The CRS describes each activity in far more detail than this, including project titles, descriptions, channels and policy markers, none of which is carried through."
    },
    {
      "transformation": "Aggregate totals",
      "description": "Activities sharing the same year, provider, recipient and dimensions are summed. Commitments, disbursements, net disbursements and grant equivalents are each aggregated as separate measures. Every figure is reported in US dollars at current prices."
    }
  ],
  "version_history": [],
  "gaps": [
    "details.description",
    "license.next_expected_update",
    "scope_and_units.countries",
    "scope_and_units.granularity",
    "scope_and_units.currencies",
    "methodology.description",
    "data_notes",
    "usage.api_access",
    "usage.python",
    "version_history"
  ],
  "document_errors": [
    {
      "member": "version_history",
      "line": 1,
      "reason": "missing_release_title"
    }
  ]
}
```

Read from the dev graph (`provenance/OECDCreditorReportingSystem`, an orphan Provenance). Its changelog headings carry a date and no title, so `version_history` is empty and `document_errors` names the first heading. `next_expected_update` is `null` until the pipeline writes `nextDataRefreshDate`.

---

### GET /api/graph/sources

The source index: every source this deployment admits data from, the Data Commons that serves it, the datasets and provenances admitted under it, and the ones it excludes. A caller can link each provenance at `GET /api/graph/documentation/{dcid}`. The route makes no graph read and has no refusal code of its own. The body is encoded once per origin catalogue `build_id` and answers `public, max-age=300` with a strong `ETag` and a gzip representation.

The index joins two things. The origin registry (`origin_catalogue/data/registry.toml`, packaged with the service) lists the reviewed base Data Commons sources and UN System Data Commons agencies, with each base source's excluded provenances and each UN agency's excluded series. The request's origin catalogue lists the provenances admitted under each source. Entries come in this order:

1. one entry per DFO source, in provenance-table order, grouping each DFO provenance under the `Source` its `source` arc names, with no exclusions;
2. each base registry entry, in registry order;
3. each UN agency, in registry order, left out when the UN origin is off (`ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED`). On the UN deployment every agency sits under one `Source`, so the agency provenance stands as the source and as its one provenance.

Each entry lists `datasets`, each `{ ref, label, provenances: [{ ref, label }] }`, in first-seen order. A provenance groups under its first `isPartOf` parent typed `Dataset`; a provenance with no Dataset forms its own group, keyed and labelled by itself, so `datasets[].ref` is then the provenance dcid. A registry entry the loaded catalogue lists no provenance under serves `datasets: []`, as when the registry is newer than the catalogue build. `provenances[].label` is the provenance table's name.

Each dataset also carries five card facts, the same values `GET /api/graph/documentation/{ref}` serves for it: `description` (`details.description`), `license_type` (`license.type`), `update_cadence` (`license.update_cadence`), `years_covered` (`scope_and_units.years_covered`) and `source_url` (`details.source_url`). The catalogue job computes them at build time from the graph (`origin_catalogue.card_facts`) and stores the licence's label and the raw `sourceReleaseFrequency` token, which this route maps to its label as the documentation route does. They follow that route's roll-up rules over the Dataset's admitted children, and an orphan provenance is its own sole child. Each is `null` where the documentation page serves `null`, and all five are `null` on UN System Data Commons entries. A card and its page agree for the graph the loaded build read. The build refreshes them at each identity change, at least weekly, so a card can lag a graph edit the page already shows. An empty `exclusions` list states that the entry was reviewed for overlap with DFO.

A base source's `exclusions` lists the provenances it holds that the deployment does not admit, each `{ ref, reason }`. `reason` is `superseded_by_dfo` for an upstream copy of a dataset DFO imports, and `superseded_by_un_system` for a provenance whose data the UN System Data Commons serves (`dc/base/UN_SDG`). The index names no successor: each answer names its own provenance and Data Commons. DFO sources and UN agencies list no exclusions, including a series a UN agency's registry entry excludes.

A provenance a base source excludes is never also listed under that source's `datasets`, even when the loaded build, older than the registry, still holds it.

```json
{
  "sources": [
    {
      "source": {
        "ref": "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
        "label": "Organisation for Economic Co-operation and Development (OECD)"
      },
      "data_commons": { "id": "dfo", "label": "Development Finance Observatory" },
      "datasets": [
        {
          "ref": "provenance/OECDCreditorReportingSystem",
          "label": "OECD Creditor Reporting System",
          "provenances": [
            {
              "ref": "provenance/OECDCreditorReportingSystem",
              "label": "OECD Creditor Reporting System"
            }
          ],
          "description": "The Creditor Reporting System (CRS) is the OECD Development Assistance Committee's activity-level database of development finance. ...",
          "license_type": "OECD Terms and Conditions",
          "update_cadence": "Quarterly",
          "years_covered": "1973-2024",
          "source_url": "https://data-explorer.oecd.org/vis?tm=crs&pg=0&snb=30&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_CRS%40DF_CRS&df[ag]=OECD.DCD.FSD&df[vs]=1.6&isAvailabilityDisabled=false&page=0"
        }
      ],
      "exclusions": []
    },
    {
      "source": { "ref": "dc/s/WorldBank", "label": "World Bank" },
      "data_commons": { "id": "base", "label": "Data Commons by Google" },
      "datasets": [
        {
          "ref": "dc/d/WorldBank_WorldDevelopmentIndicators",
          "label": "World Development Indicators",
          "provenances": [
            {
              "ref": "dc/base/WorldDevelopmentIndicators",
              "label": "World Development Indicators"
            }
          ],
          "description": "The primary World Bank collection of development indicators encompassing global economic and social data.",
          "license_type": "Creative Commons Attribution License",
          "update_cadence": "Monthly",
          "years_covered": null,
          "source_url": "https://datacatalog.worldbank.org/search/dataset/003771"
        }
      ],
      "exclusions": [
        { "ref": "dc/base/WorldBank_IDS", "reason": "superseded_by_dfo" }
      ]
    },
    {
      "source": { "ref": "dc/s/UnitedNationsUn", "label": "United Nations (UN)" },
      "data_commons": { "id": "base", "label": "Data Commons by Google" },
      "datasets": [],
      "exclusions": [{ "ref": "dc/base/UN_SDG", "reason": "superseded_by_un_system" }]
    },
    {
      "source": { "ref": "undata/p/SDG", "label": "SDG Global Database" },
      "data_commons": { "id": "un_system", "label": "UN System Data Commons" },
      "datasets": [
        {
          "ref": "undata/p/SDG",
          "label": "SDG Global Database",
          "provenances": [{ "ref": "undata/p/SDG", "label": "SDG Global Database" }],
          "description": null,
          "license_type": null,
          "update_cadence": null,
          "years_covered": null,
          "source_url": null
        }
      ],
      "exclusions": []
    }
  ]
}
```

The example is abridged. It shows four entries, cuts each `datasets` list to at most one and shortens the CRS description, while a served list names every admitted dataset under its source. Adding or removing a source is a pull request to the registry file, and `GET /api/graph/capabilities` serves `source_catalogue_curated: true`.

---

### GET /api/graph/schema

Serves the JSON Schema of every request and response body, keyed by name under `schemas`: `health_response`, `resolve_request`, `resolve_response`, `node_response`, `nodes_request`, `nodes_response`, `members_response`, `roots_response`, `source_documentation_response` and `source_catalogue_response`. `capabilities` holds the path of the capability document. The document is encoded once when the door is built, answers `public, max-age=300` with a strong `ETag` and a gzip representation, and honours `If-None-Match` with `304`. It changes only with a deploy, and `capability_version` hashes the schemas with their descriptions removed.

---

### GET /api/graph/capabilities

Returns deployment capabilities, supported input limits, and the data builds this process serves. The body answers `public, max-age=300` with an `ETag`, or `no-store` while the surface has not published its dependencies or the graph generation is unknown. Both gaps clear on a later read. `coverage_index` names the bound graph release, the build its coverage comes from.

- `capability_version` is a content hash of the limits, door liveness, refusal bodies and wire schemas. It stays put when `data_builds` changes and moves with every schema change, so the examples in this document show it as `graph_<token>`; read the running deployment's value here.
- `data_builds` names the graph generation, the origin catalogue, the coverage index and the code index in use. Each is `null` where it is unknown or absent. They change without a deploy.
- `kinds` is the closed list of 16 kinds the node route and resolve classify by, and `resolve_kinds` the ten that a resolve `kinds` filter accepts.
- `node_shape_gaps` lists the member paths the node route names in `gaps` on every node that carries the member: `stat_var.definition.measure.definition`, since no graph node holds a measure's definition.
- `source_document_gaps` lists the member paths the documentation route always empties.
- `caps` publishes the enforced limits.

The example comes from the app built with `make_client` and the restrictive test catalogue; `build` is `unknown` outside an image.

```json
{
  "capability_version": "graph_<token>",
  "build": "unknown",
  "data_builds": {
    "graph_generation": null,
    "origin_catalogue": "restrictive-catalogue-6fde3fb68115",
    "coverage_index": null,
    "code_index": "1be06f70b06f447fbc48644d99f3d023f30bf8987b2ccdf0b82c6ca73228e7e7"
  },
  "doors": [
    { "name": "resolve", "live": true },
    { "name": "node", "live": true },
    { "name": "members", "live": true },
    { "name": "roots", "live": true },
    { "name": "sources", "live": true },
    { "name": "schema", "live": true }
  ],
  "kinds": ["country", "continent", "subregion", "continental_union", "place_other", "aggregate", "organisation", "group", "stat_var", "code_system", "code", "unit", "source", "dataset", "provenance", "other"],
  "resolve_kinds": ["country", "continent", "subregion", "continental_union", "organisation", "aggregate", "place_other", "group", "code", "other"],
  "recipe_classes": [],
  "node_shape_gaps": ["stat_var.definition.measure.definition"],
  "source_document_gaps": ["scope_and_units.countries", "usage.api_access", "usage.python"],
  "source_catalogue_curated": true,
  "caps": {
    "node_pool_size": 4,
    "resolve_timeout_seconds": 30,
    "node_timeout_seconds": 10,
    "max_dcid_chars": 512,
    "max_node_batch_dcids": 50
  }
}
```

---

### GET /api/graph/healthz

Reports operational status, the running build commit, and the deployment capability version token, the same value `GET /api/graph/capabilities` serves.

```json
{"status": "ok", "build": "unknown", "capability_version": "graph_<token>"}
```

---

## Configuration

The Graph API reads environment variables during module initialization and application startup:

| Variable                        | Type    | Default                 | Description                                                                         |
| :------------------------------ | :------ | :---------------------- | :---------------------------------------------------------------------------------- |
| `DFO_GRAPH_SURFACE`             | Boolean | `false`                 | Mounts the Graph surface on the root FastAPI application (`1` or `true`).           |
| `GRAPH_DCP_BASE_URL`            | String  | _(Required if enabled)_ | Base URL for the upstream Data Commons Platform instance. The mixer serves SDMX under the same root, so the SDMX availability reads use it too. |
| `GRAPH_DCP_TOKEN`               | String  | `""`                    | Optional access token sent via `x-graph-token` header to DCP and to its SDMX root.  |
| `GRAPH_NODE_POOL_SIZE`          | Integer | `4`                     | Worker count of the DCP-read executor. It admits one read per worker and refuses the next with `503`. |
| `GRAPH_RESOLVE_TIMEOUT_SECONDS` | Integer | `30`                    | Request timeout for `/api/graph/resolve`.                                           |
| `GRAPH_NODE_TIMEOUT_SECONDS`    | Integer | `10`                    | Request timeout for `/api/graph/node/{dcid}`, `/api/graph/nodes`, `/api/graph/documentation/{dcid}`, and the label and facet reads of `/api/graph/node/{dcid}/members`. |
| `GRAPH_PROVENANCE_DOC_CACHE_SIZE` | Integer | `512`                 | Most source documents `/api/graph/documentation/{dcid}` keeps in its cache at once. |
| `GRAPH_NODE_ENTRY_CACHE_SIZE`   | Integer | `2048`                  | Most built node entries the node routes keep in their cache at once.                |
| `GRAPH_SDMX_TIMEOUT_SECONDS`    | Integer | `30`                    | HTTP timeout of the UN deployment's SDMX client the node reader uses.              |
| `UN_GRAPH_BASE_URL`             | String  | UN System Data Commons  | API root for `undata/` dcids (`https://unsd-datacommons.gcp.un-icc.cloud/core/api`), shared with every surface and the origin catalogue job. |
| `GRAPH_RELEASE_URI`             | String  | _(Required)_            | Graph release root; the code index and the coverage behind currencies, granularity and variable availability come from the release the pointer names. A missing or unopenable release fails startup. |
| `ORIGIN_CATALOGUE_URI`          | String  | _(Required)_            | Origin catalogue root; a missing or unloadable catalogue fails startup.             |

---

## Error handling

Errors use RFC 9457 problem detail envelopes:

| HTTP Status | Problem Code (`code`) | Trigger Condition                                                                                                                                      | Pointer                                                                                     | Retryable |
| :---------- | :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ | :-------- |
| `400`       | `request_invalid`     | Malformed request body, text length exceeding 200 characters, batch size outside 1–25, extra keys, or `code_system` without `code` in `kinds`.          | The offending body field, named by the platform's own request validation (e.g. `/texts/0`, `/code_system`). | No        |
| `400`       | `request_invalid`     | `dcid` path parameter exceeding 512 characters.                                                                                                        | `/dcid`                                                                                     | No        |
| `400`       | `request_invalid`     | `POST /api/graph/nodes` with `dcids` outside 1–50 entries, a malformed or over-long dcid, or extra keys.                                              | The offending body field (e.g. `/dcids/0`).                                                 | No        |
| `400`       | `request_invalid`     | `GET /api/graph/node/{dcid}/members` with a `limit` outside 1–1000, a `relation` a held node lacks, a cursor that does not decode or belongs to another walk, a `projection`, `as_of` or filter the relation does not take, or an `as_of` on an undated membership. | `/limit`, `/relation`, `/cursor`, `/projection`, `/as_of`, or the filter's name. | No        |
| `400`       | `request_invalid`     | `as_of` on `GET /api/graph/node/{dcid}` for a membership that records no dates.                                                                        | `/as_of`                                                                                    | No        |
| `404`       | `node_not_held`       | Requested DCID does not exist in the upstream DCP graph, including a `GET /api/graph/node/{dcid}/members` dcid the deployment does not hold.          | `/dcid`                                                                                     | No        |
| `400`       | `request_invalid`     | `GET /api/graph/documentation/{dcid}`'s `dcid` is held, but is neither a `Dataset` nor a `Provenance` node, including a `Source` dcid.   | `/dcid`                                                                                     | No        |
| `503`       | `service_unavailable` | `GET /api/graph/documentation/{dcid}`'s own Dataset children read came back truncated (more than 256 children, or a surviving `nextToken`).            | None                                                                                        | No        |
| `404`       | `not_found`           | Unmatched route or empty DCID path parameter (`/api/graph/node/`).                                                                                     | None                                                                                        | No        |
| `502`       | `upstream_error`      | DCP read failure with a 4xx upstream response status.                                                                                                  | None                                                                                        | Yes       |
| `503`       | `service_unavailable` | Upstream 5xx or network errors, every DCP-read worker busy, the surface still starting or shutting down, no membership policy published, or a `facets` request whose node types were not read by the deadline. | None                                                                                        | Yes       |
| `504`       | `timeout`             | Resolution, node lookup (single or batch), or source documentation read exceeding configured deadline.                                                 | None                                                                                        | Yes       |
| `500`       | `internal_error`      | Unhandled internal or executor failures.                                                                                                               | None                                                                                        | No        |

The 502, 503, and 504 refusals describe an infrastructure outcome with no request field at fault, and carry no pointer.
