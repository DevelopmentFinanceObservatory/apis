The Graph API surface provides entity resolution and node inspection over Data Commons Platform (DCP) graphs. It resolves free-text geographical mentions into graph-confirmed DCIDs, expands hierarchical containment for subregions and continents, extracts typed node metadata for statistical variables, provenance sources, geographical entities, and schema nodes, and assembles a per-source documentation document rolled up from a Dataset's own Provenance children.

The surface mounts at `/api/graph` and operates as an independent composition root within `dfo-apis`.

## Architecture

```
                    +------------------------------------------+
                    |           POST /api/graph/resolve        |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |         entity_resolution.EntityResolver |
                    |   (resolvekit match, gate, fallback,     |
                    |    graph confirmation -- one component)  |
                    +------------------------------------------+
                                         | Resolved / Ambiguous /
                                         | NotHeld / NotFound
                                         v
+--------------------------------+       |
|    GET /api/graph/node/{dcid}  |       | Region expansion
+--------------------------------+       | (continent/subregion only)
               |                         |
               v                         v
+--------------------------------------------------------------+
|                    GraphNodeReader (DCP)                     |
|           Bounded thread pool + Non-blocking semaphore       |
+--------------------------------------------------------------+
```

### Components

- **Entity resolution (`entity_resolution`)**: Free-text resolution and graph confirmation both run inside the shared `entity_resolution.EntityResolver` component (`packages/dfo-data-tools/src/entity_resolution`), acquired once per process (`entity_resolution.acquire_shared()`) and shared with Search when both are mounted in the same deployment. It resolves over `resolvekit`'s four bundled country-level geo modules plus `dfo_graph`, a custom pack built from this deployment's own graph (`entity_resolution.MODULE_IDS`). Both the bundled packs and `dfo_graph` are baked into the image at build time (`scripts/warm_resolvekit_cache.py`), so no pack is built or composed at runtime. Because the SQLite engine underlying `resolvekit` relies on thread affinity, all matching queries run on that component's own dedicated single-worker thread; graph confirmation runs on this surface's own bounded DCP-read executor (below). This door's own job is limited to mapping the component's result onto the wire and expanding a continent or subregion winner to its member countries through graph containment.
- **DCP Node Reader (`GraphNodeReader`)**: Graph confirmation reads (inside the component), this door's own region-expansion reads, and the node door's reads (`GET /api/graph/node/{dcid}` and `POST /api/graph/nodes`) all execute against the upstream Data Commons Platform instance through one bounded thread pool executor (`GRAPH_NODE_POOL_SIZE`).
- **Admission Control**: Read access to DCP uses a non-blocking admission semaphore matching the thread pool capacity. When saturated, incoming reads fail immediately with `503 Service Unavailable` (`service_unavailable`), allowing clients to back off and retry.
- **Contract Schema and Refusal Typing**: Responses conform to fixed Pydantic models. Error envelopes use RFC 9457 problem details.
- **Source Documentation Reader (`materialise.provenance_doc`)**: `GET /api/graph/documentation/{dcid}` submits one whole `read_provenance_doc` call to the same bounded DCP-read executor and admission semaphore, since its own five reads are each dependent on the last. Also fills `scope_and_units.years_covered`, rolled up from the same batched read; `scope_and_units.countries` has no graph answer at all -- see the endpoint's own section below.

---

## Endpoints

Every response, success or refusal, carries `x-dfo-trace-id`, `x-dfo-commit`, and a `Server-Timing: total` header. Graph success bodies carry no body-level trace id; the header is the channel.

### POST /api/graph/resolve

Resolves a batch of 1 to 25 free-text strings to graph-confirmed entities through the shared `entity_resolution` component -- countries, continents, UN subregions, continental unions, organisations, DAC aggregates, and membership groups (for example G7, G20, DAC, the EU, OECD) are all in scope; `geo.region` (loose undata-geo groupings such as WHO member-state lists) is excluded. A `resolved`/`resolved_region`/`resolved_group`/`ambiguous` outcome names an entity, or entities, that the input text itself names, by a listed name, alias, or code. A spelling-corrected match, for example "Lagos" resolving to Laos, comes back as `not_found` instead, with the corrected name as the first near match. Resend that name, or a name from the list.

A text that resolves to nothing on the first pass gets one retry, with a leading "the" stripped (case-insensitive, only when followed by whitespace) and any trailing run of `.`, `,`, `;`, `:`, `!`, or `?` characters stripped, then the same resolve-and-gate pass runs again. The retry's result replaces the miss only when it produces a survivor; a text that already resolves on the first pass never reaches it. "the EU" resolves to the European Union this way, and "Kenya." resolves to `country/KEN`. A name that legitimately carries the article, such as "the Gambia", resolves on the first pass and is untouched by the retry.

Every membership group's own members are resolved by `entity_resolution` itself, not by this door: `resolved_group` carries them straight from the component's own graph confirmation, with no further graph read here. This door has no `role_property` (`entity_resolution.Admission.admits` checks a resolved entity's `graph_type` against one, e.g. `observationAbout` or `financingProvider`) to decide with, so a group carrying both an entity reading and members -- the EU is the only case seen so far -- returns both. `entity`'s presence, even one that type-admits for a caller's own role, does not imply it carries data: a group's own node commonly has none while its members do. A generic caller should default to `members` and treat `entity` as an additional reading, not a preferred one.

A text with no listed-name survivor takes the description-arc fallback ladder, which admits a node only when the text is one of that node's own listed names, so a description-arc match the text does not name is not a resolution either.

`resolved_group.members` has no enforced bound of its own: a group resolves to its full membership, drawn from a closed offline set of 33 curated groups (resolvekit's bundled `geo.*` modules plus `dfo_graph`), the largest being the United Nations at 193. This is not configurable and no request parameter narrows it. `search.engine`'s own `SEARCH_MAX_CHILD_PLACES`/`SEARCH_MAX_STANDARD_PROBE_PLACES` bound (256) is a separate `search.engine` constant that does not apply to this door. "G20" resolves to its 19 country members; that is correct and intended, not a gap -- this door does not add the EU or the African Union to it.

#### Request Constraints

- `texts`: Array of 1 to 25 strings. Each string is stripped of leading and trailing whitespace. The stripped string must be non-empty and at most 200 characters.
- Extra fields are forbidden (`extra="forbid"`).

#### Resolution Pipeline

1. **Component Resolve**: `entity_resolution.EntityResolver.resolve(texts)` resolves the whole batch in one call -- resolvekit matching, the listed-name gate, the description-arc fallback ladder, and graph confirmation all run inside this one call.
2. **Region Expansion**: Every text that resolved to a continent or subregion expands to its member countries through graph containment -- one batched `child_places_batch(..., child_type="Country")` call for the whole request, then one batched `node_arcs_many(..., properties=("name",))` call for the members' display names. Every member is a graph node by construction, so there is no separate "member not held" case.
3. **Result Assembly**: Maps each resolution into a typed status object, index-aligned with the input batch.

#### Result Grammar

All batch outcomes return HTTP 200 with an entry corresponding to each input string. `entity_type` is the graph's own `typeOf` value for the entity -- open and graph-sourced, any value the graph reports, not a fixed vocabulary. `kind` is `entity_resolution`'s own coarser classification, closed to eight values:

| Status            | Condition                                                                                                                                 | Payload Structure                                                                                         |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------- |
| `resolved`        | Single graph-confirmed entity the input text itself names.                                                                               | `entity`: `{ dcid, name, entity_type, kind }`                                                             |
| `resolved_region` | Continent or subregion expanded to its confirmed member countries.                                                                       | `region`: `{ dcid, name, entity_type, kind }`, `members`: `[{ dcid, name }]`                              |
| `resolved_group`  | Membership group (for example G7, G20, DAC, EU, OECD) expanded to its graph-confirmed members.                                            | `entity`: `{ dcid, name, entity_type, kind }` or `null`, `members`: `[{ dcid, name, entity_type, kind }]` |
| `ambiguous`       | Two or more graph-confirmed candidates the input text itself names.                                                                       | `candidates`: `[{ dcid, name, entity_type, kind }]`                                                       |
| `not_found`       | No candidate the input text itself names survived; a spelling-corrected match lands here too, its corrected name as the first near match. | `near_matches`: `[name, ...]`                                                                              |
| `not_held`        | A real entity was resolved (or narrowed to), but this deployment's graph does not hold it.                                                | `name`, `dcid`                                                                                             |

`resolved_region`'s own node IS the region it names, so that field is called `region` and is never `null`. `resolved_group`'s `entity` is different: a group is not itself a held node the way a region is, so `entity` is one of two readings this door reports alongside `members` (the group's own node when one exists, `null` when it does not) -- the names differ because what they hold differs, not by oversight.

`status` is a closed union of the six values above, and this door sends nothing outside it. A client that mirrors this contract by hand, rather than through a generated type, should still treat a `status` it does not recognise as an explicit, surfaced failure: never render it as an empty result, and never fold it into `not_found` or `not_held` just because it resembles one.

`kind` values, one line each:

| `kind`              | Meaning                                                                                               |
| :------------------ | :---------------------------------------------------------------------------------------------------- |
| `country`           | A Country-typed graph node.                                                                           |
| `continent`         | A Continent-typed graph node; expands to member countries (`resolved_region`).                        |
| `subregion`         | A UN geographic subregion (e.g. Sub-Saharan Africa); expands to member countries (`resolved_region`). |
| `continental_union` | A continental political or economic union (e.g. the African Union).                                   |
| `organisation`      | An Organization-typed graph node (e.g. a development bank).                                           |
| `aggregate`         | A DAC recipient-unallocated-by-country bucket, not a real place.                                      |
| `place_other`       | A subnational place: a state, city, or administrative area.                                           |
| `other`             | None of the above.                                                                                    |

Only `continent` and `subregion` expand to member countries through `resolved_region`; `continental_union` sits next to them in the `kind` vocabulary but comes back as a plain `resolved` entity, or as `resolved_group.entity`, never `resolved_region`. A membership group (G7, G20, DAC, OECD, the EU) is not a `kind` value at all -- it never confirms as an entity in its own right, so it carries no `kind` of its own and surfaces only through `resolved_group.members`' own `kind` (always `country` for every group seen so far) and, where one exists, `resolved_group.entity`. `resolved_group.members` is never empty: a group with no graph-confirmed member, and no entity reading either, resolves `not_held` instead of an empty `resolved_group`. `ambiguous.candidates` is ordered by confidence descending, then by `dcid` -- a caller that auto-picks `candidates[0]` is picking the most confident match. `not_found.near_matches` is capped at 3 entries. `not_held.dcid` identifies the real entity `entity_resolution` resolved, but that `dcid` does not open on `GET /api/graph/node/{dcid}` either (that door reports `404 node_not_held` for it too) -- a caller's own move here is the same as for `not_found`: report the gap, don't retry.

#### Request Example

```json
{
  "texts": ["Kenya", "East Africa", "Atlantis", "World Bank", "G7"]
}
```

#### Response Example

```json
{
  "results": [
    {
      "status": "resolved",
      "entity": {
        "dcid": "country/KEN",
        "name": "Kenya",
        "entity_type": "Country",
        "kind": "country"
      }
    },
    {
      "status": "resolved_region",
      "region": {
        "dcid": "EasternAfrica",
        "name": "Eastern Africa",
        "entity_type": "UNGeoRegion",
        "kind": "subregion"
      },
      "members": [
        { "dcid": "country/KEN", "name": "Kenya" },
        { "dcid": "country/UGA", "name": "Uganda" },
        { "dcid": "country/TZA", "name": "Tanzania" }
      ]
    },
    {
      "status": "not_found",
      "near_matches": []
    },
    {
      "status": "ambiguous",
      "candidates": [
        {
          "dcid": "org/IBRD",
          "name": "International Bank for Reconstruction and Development",
          "entity_type": "Organization",
          "kind": "organisation"
        },
        {
          "dcid": "org/IDA",
          "name": "International Development Association",
          "entity_type": "Organization",
          "kind": "organisation"
        }
      ]
    },
    {
      "status": "resolved_group",
      "entity": null,
      "members": [
        { "dcid": "country/CAN", "name": "Canada", "entity_type": "Country", "kind": "country" },
        { "dcid": "country/DEU", "name": "Germany", "entity_type": "Country", "kind": "country" },
        { "dcid": "country/FRA", "name": "France", "entity_type": "Country", "kind": "country" },
        { "dcid": "country/GBR", "name": "United Kingdom", "entity_type": "Country", "kind": "country" },
        { "dcid": "country/ITA", "name": "Italy", "entity_type": "Country", "kind": "country" },
        { "dcid": "country/JPN", "name": "Japan", "entity_type": "Country", "kind": "country" },
        { "dcid": "country/USA", "name": "United States", "entity_type": "Country", "kind": "country" }
      ]
    }
  ]
}
```

Live-verified against the dev graph: "G7" resolves this way, `entity: null` -- this deployment's graph holds no node for `groups/G7` itself, only its 7 member countries. "OECD" resolves the same shape with 38 members and `entity: null` too. "EU" is the one group seen so far that carries both readings -- its own held node confirms alongside its 27 members. The first 2 of its 27 members are shown below; the shape is otherwise identical to `members` above:

```json
{
  "status": "resolved_group",
  "entity": {
    "dcid": "EuropeanUnion",
    "name": "European Union",
    "entity_type": "ContinentalUnion",
    "kind": "continental_union"
  },
  "members": [
    { "dcid": "country/AUT", "name": "Austria", "entity_type": "Country", "kind": "country" },
    { "dcid": "country/BEL", "name": "Belgium", "entity_type": "Country", "kind": "country" }
  ]
}
```

---

### GET /api/graph/node/{dcid}

Fetches node metadata and relationships for a specific DCID. Multi-segment DCIDs containing slashes (such as `country/KEN` or `dc/base/UNSD_M49_SubRegion`) are supported via a custom URL path convertor matching `[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*`.

The `dcid` path parameter is capped at 512 characters, published as `caps.max_dcid_chars` on GET /api/graph/capabilities -- the same cap `GET /api/graph/documentation/{dcid}` enforces on its own `dcid`. A longer value is refused `400 request_invalid` before any DCP read runs.

#### Property Allowlist

Reads retrieve a fixed property allowlist to avoid oversized properties such as polygon geometries:

```python
(
    "typeOf", "name", "description", "definition", "populationType",
    "measuredProperty", "observationProperties", "constraintProperties",
    "license", "licenseType", "source", "sourceReleaseFrequency",
    "automaticRefreshType", "url", "sourceDataUrl", "descriptionUrl",
    "dataTransformationLogic", "lastDataRefreshDate", "nextDataRefreshDate",
    "earliestObservationDate", "latestObservationDate", "isPartOf",
    "provenanceCategory", "containedInPlace"
)
```

#### Shape Classification

The node is classified by its `typeOf` arc in order of precedence:

| Shape (`shape`) | Triggering `typeOf` Arc                        | Key Fields                                                                                                                                                                                                                                                                                                                                                                                 |
| :-------------- | :--------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stat_var`      | `StatisticalVariable`                          | `dcid`, `label`, `definition`, `description`, `roles`, `population_type`, `source_indicator_codes`, `data_commons`: `[{ id, label }]`, `facets` (null), `processing_level` (null), `sources` (null)                                                                                                                                                                                                        |
| `source`        | `Provenance` or `Source`                       | `dcid`, `label`, `license`, `license_type`, `description`, `url`, `source_data_url`, `description_url`, `transformation_logic_url`, `last_data_refresh_date`, `next_data_refresh_date`, `earliest_observation_date`, `latest_observation_date`, `publisher`: `{ dcid, label }`, `dataset`: `{ dcid, label }`, `freshness` (`release_frequency`, `release_frequency_label`, `refresh_type`), `data_commons`: `{ id, label }` or null |
| `entity`        | Place types (`Country`, `City`, `State`, etc.) | `dcid`, `label`, `entity_type`, `contained_in`: `[{ dcid, label }]`                                                                                                                                                                                                                                                                                                                        |
| `node`          | All other node types                           | `dcid`, `label`, `types`, `properties`: `{ [property]: [values] }`                                                                                                                                                                                                                                                                                                                         |

For `entity` shapes, an auxiliary DCP query fetches labels for parent entities in `containedInPlace`. For `source` shapes, that same auxiliary query fetches labels for `publisher` (the `source` arc's target) and `dataset` (the `isPartOf` arc's target), batched into one call. The allowlist read and the auxiliary query run as one executor job under one admission slot, sharing one deadline. `source.label` prefers `dataset.label`, falling back to the node's own `name` arc when it has no parent Dataset -- a `Provenance`'s own `name` is an import identifier (e.g. `GeoBoundaries_Jordan`), not a display title. A `name` value equal to its own dcid (a junk triple some imports write) is never served as a label.

A `stat_var`'s `source_indicator_codes` lists every value of its `sourceIndicatorCode` arc: the codes naming the source series the variable comes from, as the publisher codes it, and `[]` when the node carries none. IMF WEO GDP (`ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal`) lists `NGDP` and `NGDPD`. The codes identify no variable, so Search and Datasets never treat them as a constraint.

`data_commons` names the Data Commons a fact comes from, read from the request's origin catalogue with no graph read. Its `id` is `dfo`, `base` or `un_system`, and its `label` is Development Finance Observatory, Data Commons by Google or UN System Data Commons. A `stat_var` lists every origin holding admitted data for the variable, DFO first, then base, then the UN, and `[]` when the catalogue admits none; `Count_Person` lists both `dfo` and `base`, and a variable no catalogue table lists counts as `base`, as search admits it. A `source` names the origin of an admitted provenance, and serves `null` for a provenance the catalogue does not admit, a `Dataset` or a `Source`. `entity` and `node` shapes carry no `data_commons`, since a place or a schema node is shared by every Data Commons. A dcid under `undata/` is read from the UN System Data Commons.

#### Response Example (`stat_var`)

```json
{
  "shape": "stat_var",
  "dcid": "Count_Person",
  "label": "Total population",
  "definition": "mp=count,pt=Person",
  "description": "The total number of people in a population.",
  "roles": ["observationAbout"],
  "population_type": "Person",
  "source_indicator_codes": [],
  "data_commons": [
    { "id": "dfo", "label": "Development Finance Observatory" },
    { "id": "base", "label": "Data Commons by Google" }
  ],
  "facets": null,
  "processing_level": null,
  "sources": null
}
```

#### Response Example (`source`)

```json
{
  "shape": "source",
  "dcid": "dc/base/WorldDevelopmentIndicators",
  "label": "World Development Indicators",
  "license": "https://datacatalog.worldbank.org/search/dataset/0037712",
  "license_type": "CreativeCommonsAttributionLicense",
  "description": "The primary World Bank collection of development indicators encompassing global economic and social data.",
  "url": "https://datatopics.worldbank.org/world-development-indicators/",
  "source_data_url": "http://api.worldbank.org/v2/en/country/AFG?downloadformat=csv",
  "description_url": "https://datacatalog.worldbank.org/search/dataset/0037712",
  "transformation_logic_url": "https://github.com/datacommonsorg/data/tree/master/scripts/world_bank/wdi/",
  "last_data_refresh_date": null,
  "next_data_refresh_date": null,
  "earliest_observation_date": null,
  "latest_observation_date": null,
  "publisher": { "dcid": "dc/s/WorldBank", "label": "World Bank" },
  "dataset": {
    "dcid": "dc/d/WorldBank_WorldDevelopmentIndicators",
    "label": "World Development Indicators"
  },
  "freshness": {
    "release_frequency": "P1M",
    "release_frequency_label": "Monthly",
    "refresh_type": "GCPCloudRunAutomaticRefresh"
  },
  "data_commons": { "id": "base", "label": "Data Commons by Google" }
}
```

Live-verified against the dev graph (`dc/base/WorldDevelopmentIndicators`). `label` comes from the parent Dataset's own name, not the provenance's own `name` arc value (`WorldDevelopmentIndicators`, an import identifier).

---

### POST /api/graph/nodes

Reads up to 50 nodes in one request, under one admission slot. A client that needs several nodes at once sends them here: parallel `GET /api/graph/node/{dcid}` calls each take a slot of the shared pool (`GRAPH_NODE_POOL_SIZE`, 4 by default), so the fifth concurrent read is refused `503 service_unavailable`.

The door makes one allowlisted arc read for every distinct dcid, then one batched name read for the label targets of every held `source` and `entity` node, skipped when none has a target. Each entry is the body `GET /api/graph/node/{dcid}` returns for that dcid, so shape classification, labels, and `data_commons` follow that route's rules above. A dcid the graph does not hold gets a `not_held` entry where the GET route answers `404 node_not_held`.

#### Request constraints

- `dcids`: array of 1 to 50 dcids, published as `caps.max_node_batch_dcids` on GET /api/graph/capabilities. Each matches the GET route's path syntax (`[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*`) and its 512-character ceiling. A list outside these bounds, a malformed dcid, or an extra key is refused `400 request_invalid` before any DCP read runs.

```json
{ "dcids": ["country/NGA", "Count_Person", "country/XYZ"] }
```

#### Response

`nodes` is index-aligned with `dcids`, and a repeated dcid repeats its entry.

```json
{
  "nodes": [
    { "shape": "entity", "dcid": "country/NGA", "label": "Nigeria", "entity_type": "Country", "contained_in": [{ "dcid": "africa", "label": "Africa" }] },
    { "shape": "stat_var", "dcid": "Count_Person", "label": "Population", "...": "..." },
    { "shape": "not_held", "dcid": "country/XYZ" }
  ]
}
```

---

### GET /api/graph/documentation/{dcid}

Assembles one source's documentation document, in the field names a consuming frontend design specifies. `dcid` accepts either a `Dataset` dcid (the catalogue's own grain -- see below) or a `Provenance` dcid: a Provenance with a parent Dataset climbs to it and rolls up across every one of that Dataset's own children, not just the given node alone; a Provenance with no parent (an orphan -- e.g. the ONE-minted `provenance/OECDCreditorReportingSystem`) is treated as its own single child. A `Source` dcid (the publishing organisation, e.g. `dc/s/WorldBank`) is refused `400 request_invalid`: this door documents a publication, not a publisher. A UN System Data Commons agency provenance (`undata/p/SDG`) is read from the UN deployment and documented the same way.

The graph's own three-level chain is **Source** (the publishing organisation) -> **Dataset** (the publication, e.g. "World Development Indicators") -> **Provenance** (one import). Several Provenance nodes often share one Dataset -- two ACS subject tables, two GeoBoundaries country imports -- so the catalogue this door serves lists Dataset nodes, and the documentation document is assembled by rolling up the Provenance nodes beneath one.

#### Roll-up rules

- A Dataset's children are the `Provenance` nodes that name it in `isPartOf` and that the request's origin catalogue admits. A withheld or excluded child counts toward no member. A Dataset with no admitted child answers `404 node_not_held`, since the source index lists no entry for it. A Provenance's parent Dataset is its first `isPartOf` parent typed `Dataset`. A Provenance whose parents are all something else (a Source, a `ProvisionalNode`, a node the graph does not hold) is an orphan.
- `details.description`, `details.source_url`, `license.type`, `license.update_cadence` and `scope_and_units.years_covered` come from `origin_catalogue.card_facts.card_facts_of`, the function the catalogue job computes the source index's card facts with (see `GET /api/graph/sources`).
- `details.name_of_source`, `details.source_url`, `details.publisher` read from the Dataset node directly, no roll-up. `source_url` falls back to the sole child's own `url` when the Dataset has exactly one child and none of its own. `name_of_source` falls back to the given node's own `name` for an orphan, or when the resolved Dataset carries none; both reads apply the same self-referential-name guard `publisher` uses on the plain node door (above) -- a `name` value equal to its own dcid is never served, falling through to the next name in the chain instead.
- `details.source` and `details.data_commons` come from the request's origin catalogue. `source` is the provenance table's source (`{ ref, label }`, the registry's name for the `Source` node; for a UN agency, the agency provenance itself) and `data_commons` its Data Commons (`{ id, label }`). A Provenance dcid uses its own catalogue row. A Dataset dcid uses its admitted children's rows and serves each when they share one value; otherwise it is `null` and gapped, so a Dataset whose children sit under two sources serves `details.source` `null`.
- `license.last_update` is the max across every value every child asserts for `lastDataRefreshDate`, flattened across children; `license.next_expected_update` is the min across `nextDataRefreshDate`, same flattening. A child that asserts the same property twice contributes both values, not just whichever the graph returned first.
- `license.type` and `license.update_cadence` serve the value only when every child asserts exactly one value for the property and all children's values agree. A child that carries no value at all empties the field the same as an outright cross-child conflict does -- one declaring child does not speak for a publication the rest are silent on, and the path joins `gaps`. `license.type` is the label of the agreed `licenseType` node, its `name` in the graph (e.g. `Creative Commons Attribution License` for `CreativeCommonsAttributionLicense`), or its dcid when the graph holds no name for it (`origin_catalogue.card_facts.licence_labels_of`). One further node read fetches the name. `license.update_cadence` converts `sourceReleaseFrequency`'s own ISO 8601 duration token the same way `freshness.release_frequency_label` already does on the plain node door (`materialise.cadence`).
- `details.citation` (`licenseAttribution`), `details.source_data_url` (`sourceDataUrl`, the file the pipeline downloads) and `license.url` (`license`, the licence text on the publisher's site) follow the same every-child-agrees rule as `license.type` and serve the stored value unchanged. Datasets carry no `sourceDataUrl` or `license` of their own, so both come from the children. `details.source_url` is the publisher's landing page.
- `details.description` is `null` on any Dataset with more than one child -- no honest roll-up exists. A single-child Dataset (including an orphan's own degenerate one-child case) serves that one child's `description`.
- `scope_and_units.years_covered` is the earliest `earliestObservationDate` across every child that carries one, joined to the latest `latestObservationDate` across every child that carries one, as `"{start}-{end}"` (e.g. `"1973-2024"`). `null` when no child carries either property -- present on about 6.7% of provenances in the dev graph's own random sample. `scope_and_units.countries` is permanently `[]`: no graph structure answers "which countries does this source cover" for a general source -- see `materialise.provenance_doc`'s own docstring, "On scope_and_units.countries", for the evidence and what would unblock it.
- `scope_and_units.currencies` lists, sorted, the ISO 4217 code of every currency (e.g. `["EUR", "USD"]`) that a coverage series' facet carries as its `Currency` unit, across all of the document's Provenances (the given orphan, or every child of the Dataset). It reads `series_coverage.CoverageIndex.provenance_currencies`. Only facets the coverage rows use count, the same series `granularity` reads, so facet metadata no series uses adds no currency. A Provenance with no currency-valued series or no coverage rows adds none. When no Provenance carries a currency, the list is `[]` and the path `scope_and_units.currencies` is gapped.
- `scope_and_units.granularity` is computed from the same coverage index (`series_coverage.CoverageIndex.provenance_shapes`), e.g. `"Annual, by provider and recipient"`. It is served when every one of the document's Provenances has coverage series and together they share exactly one observation period that `materialise.cadence` labels (`P1Y` reads `Annual`) and the union of their variables' role axes is the flow pair, origin and destination, named by `materialise.roles.axis_noun` in that order. The union is the finest granularity, so a Provenance with one-role totals beside by-creditor variables reads as a two-axis source. A Provenance without coverage rows, two periods, a series without a period, an unlabelled period token, a union of one axis, or a role outside the two flow axes serves `null`.
- Both coverage facts see only role-bearing series. The coverage job keeps series whose variable declares one or two `observationProperties` roles: a two-role series with its donor and recipient, and a one-role series with its entity as recipient and no donor. It drops series keyed by `observationAbout` alone. A Provenance without coverage rows adds no currency and makes `granularity` `null`. A Provenance that holds both kinds is described by its role-bearing series only, so a currency or period that appears only in its `observationAbout` series is not seen.
- When `COVERAGE_ARTIFACT_URI` is unset, the `coverage` extra is missing, or the artifact cannot be read at startup, currencies serve `[]` and granularity serves `null` with no per-request log line, since startup already logged the cause. Both facts come from one coverage index snapshot per response, so a background refresh never mixes two builds in one document. A failure reading that snapshot logs one WARNING naming the Provenance dcids and serves both facts empty. A failure deriving one fact logs a WARNING naming the Provenance dcids and serves only that fact empty. The request succeeds in every case.

#### Documents

`transformation_log`, `version_history`, `methodology.description` and `data_notes` come from four Markdown documents on the sole child Provenance: `dataTransformationDescription`, `dataTransformationChangelog`, `sourceMethodologyDescription` and `dataNotes`. The pipeline stores each document MCF-escaped; this door decodes it and parses it against its template with `dfo_provenance_docs`, whose README states the templates. A Dataset with several children serves all four empty, with no error. Text members carry CommonMark, with a paragraph's soft-wrapped lines joined by one space.

- `transformation_log` has one entry per `##` step, with `transformation` the heading and `description` the step's body.
- `version_history` has one entry per `## YYYY-MM-DD <title>` release, newest first. Each has a `date`, a `title`, and `changes`, and each change has a `kind` (`added`, `changed`, `deprecated`, `removed`, `fixed` or `security`, from its `###` group) and a `note`.
- `methodology.description` is the whole methodology document.
- `data_notes` has one entry per bullet of the notes document.

A stored document longer than six characters per byte of `dfo_provenance_docs.MAX_DOCUMENT_BYTES` (the longest escaped form of the largest document the grammar accepts) is refused before decoding, with reason `document_too_large` and `line` `null`. A document that is refused, fails to decode or breaks its template, or a node carrying two distinct values of one document property, leaves its member empty and named in `gaps`, and adds one `document_errors` entry: `member` (the wire path), `line` (1-based in the decoded document, or `null`), and `reason` (a `dfo_provenance_docs.DocumentErrorReason` value, or `conflicting_values`). The rest of the document is still served. `document_errors` is always present, `[]` when every document parsed.

#### `usage`

`usage.api_access`/`usage.python` are always `null` (`graph_api.usage.usage_document`), for every `dcid`: no request shape this deployment exposes -- the mint door's `source_codes` kind, a graph read, or the SDMX bridge -- can turn a source dcid alone into a StatVar, and without one there is nothing for a `code` sample to name. Live-verified 2026-09-10: `GET /sdmx/v3/data/dataflow/DC/DF_OBS/1.0.0/*` and `GET /sdmx/v3/availability/dataflow/DC/DF_OBS/1.0.0/*/variableMeasured`, both carrying only a `c[provenance]` filter and no `c[variableMeasured]`, each return `400 {"code":3,"message":"missing required SDMX component filter variableMeasured"}` against the dev mixer. See `graph_api.usage`'s own docstring for the full investigation.

#### Reads

A document costs at most five bounded reads: the node's identity arcs, its parent Dataset when the given node is a Provenance with one, the Dataset's own children via the reverse `isPartOf` arc filtered to `{typeOf:Provenance}`, one batched read covering the publisher's own name and every child's own roll-up properties, and the name of the agreed `licenseType` node when every child names the same one. An orphan Provenance is read in that batch as its own sole child. The batch requests `description`, `url` and the four document properties only when there is exactly one child, and `name` only when there is a publisher, so a many-child Dataset never transfers them. Live-verified against `dc/d/UsCensusBureau_AmericanCommunitySurveyAcs`: 61 Provenance children returned in one page, no `nextToken`.

#### `gaps`

`gaps` and `document_errors` sit outside the consuming design's own eight members. `gaps` lists every member path that is `null`/`[]` in this particular response, by dotted path, in the design's declaration order (e.g. `"details.citation"`, `"scope_and_units.countries"`). A scalar this door cannot answer serves `null`, and an absent array serves `[]`. `scope_and_units.countries` and both `usage` entries ship `null`/`[]` for every source: countries has no source in this reader, and the `usage` pair for the reason `#usage` above records. These unconditional gaps are published on `GET /api/graph/capabilities` as `source_document_gaps`, so a caller can discover them without fetching a live document. Every other member is named in `gaps` only for the source it came back empty for.

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

Live-verified against the dev graph (`provenance/OECDCreditorReportingSystem`, an orphan Provenance). Its changelog headings carry a date and no title, so `version_history` is empty and `document_errors` names the first heading. `next_expected_update` is `null` until the pipeline writes `nextDataRefreshDate`.

---

### GET /api/graph/sources

The source index: every source this deployment admits data from, the Data Commons that serves it, the datasets and provenances admitted under it, and the ones it excludes. A caller can link each provenance at `GET /api/graph/documentation/{dcid}`. The route makes no graph read and has no refusal code of its own.

The index joins two things. The origin registry (`origin_catalogue/data/registry.toml`, packaged with the service) lists the reviewed base Data Commons sources and UN System Data Commons agencies, with each base source's exclusions. The request's origin catalogue lists the provenances admitted under each source. Entries come in this order:

1. one entry per DFO source, in provenance-table order, grouping each DFO provenance under the `Source` its `source` arc names, with no exclusions;
2. each base registry entry, in registry order;
3. each UN agency, in registry order, left out when the UN origin is off (`ORIGIN_CATALOGUE_UN_SYSTEM_ENABLED`). On the UN deployment every agency sits under one `Source`, so the agency provenance stands as the source and as its one provenance.

Each entry lists `datasets`, each `{ ref, label, provenances: [{ ref, label }] }`, in first-seen order. A provenance groups under its first `isPartOf` parent typed `Dataset`; a provenance with no Dataset forms its own group, keyed and labelled by itself, so `datasets[].ref` is then the provenance dcid. A registry entry the loaded catalogue lists no provenance under serves `datasets: []`, as when the registry is newer than the catalogue build. `provenances[].label` is the provenance table's name.

Each dataset also carries five card facts, the same values `GET /api/graph/documentation/{ref}` serves for it: `description` (`details.description`), `license_type` (`license.type`), `update_cadence` (`license.update_cadence`), `years_covered` (`scope_and_units.years_covered`) and `source_url` (`details.source_url`). The catalogue job computes them at build time from the graph (`origin_catalogue.card_facts`) and stores the licence's label and the raw `sourceReleaseFrequency` token, which this route maps to its label as the documentation route does. They follow that route's roll-up rules over the Dataset's admitted children, and an orphan provenance is its own sole child. Each is `null` where the documentation page serves `null`, and all five are `null` on UN System Data Commons entries. A card and its page agree for the graph the loaded build read. The build refreshes them at each identity change, at least weekly, so a card can lag a graph edit the page already shows. An exclusion is `{ ref, reason }`; its `reason` is `superseded_by_dfo`, for an upstream copy of a dataset DFO imports. An empty `exclusions` list states that the entry was reviewed for overlap with DFO.

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
      "datasets": [
        {
          "ref": "dc/d/UnitedNationsUn_SdgIndicatorsDatabase",
          "label": "SDG Indicators Database",
          "provenances": [{ "ref": "dc/base/UN_SDG", "label": "SDG Indicators Database" }],
          "description": "Data tracking global progress towards the United Nations Sustainable Development Goals (SDGs).",
          "license_type": "PublicDomain",
          "update_cadence": null,
          "years_covered": null,
          "source_url": "https://unstats.un.org/sdgs/dataportal"
        }
      ],
      "exclusions": []
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

The example is abridged. It shows four entries, cuts each `datasets` list to at most one, and shortens the CRS description, while a served list names every admitted dataset under its source. Adding or removing a source is a pull request to the registry file, and `GET /api/graph/capabilities` serves `source_catalogue_curated: true`.

---

### GET /api/graph/capabilities

Returns deployment capabilities, supported input limits, and active service limits.

```json
{
  "capability_version": "graph_a1b2c3d4e5f67890",
  "build": "2618ed596fb472588307ca2a7fc59d183a3f311c",
  "doors": [
    { "name": "resolve", "live": true },
    { "name": "node", "live": true },
    { "name": "sources", "live": true }
  ],
  "recipe_classes": [],
  "node_shape_gaps": {
    "facets": false,
    "sources": false,
    "processing_level": false
  },
  "source_document_gaps": [
    "scope_and_units.countries",
    "usage.api_access",
    "usage.python"
  ],
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

Reports operational status, the running build commit, and the deployment capability version token.

```json
{
  "status": "ok",
  "build": "2618ed596fb472588307ca2a7fc59d183a3f311c",
  "capability_version": "graph_a1b2c3d4e5f67890"
}
```

---

## Configuration

The Graph API reads environment variables during module initialization and application startup:

| Variable                        | Type    | Default                 | Description                                                                         |
| :------------------------------ | :------ | :---------------------- | :---------------------------------------------------------------------------------- |
| `DFO_GRAPH_SURFACE`             | Boolean | `false`                 | Mounts the Graph surface on the root FastAPI application (`1` or `true`).           |
| `GRAPH_DCP_BASE_URL`            | String  | _(Required if enabled)_ | Base URL for upstream Data Commons Platform instance.                               |
| `GRAPH_DCP_TOKEN`               | String  | `""`                    | Optional access token sent via `x-graph-token` header to DCP.                       |
| `GRAPH_NODE_POOL_SIZE`          | Integer | `4`                     | Concurrency limit for DCP-read executor and admission semaphore.                    |
| `GRAPH_RESOLVE_TIMEOUT_SECONDS` | Integer | `30`                    | Request timeout for `/api/graph/resolve`.                                           |
| `GRAPH_NODE_TIMEOUT_SECONDS`    | Integer | `10`                    | Request timeout for `/api/graph/node/{dcid}`, `/api/graph/nodes`, and `/api/graph/documentation/{dcid}`. |
| `UN_GRAPH_BASE_URL`             | String  | UN System Data Commons  | API root for `undata/` dcids (`https://unsd-datacommons.gcp.un-icc.cloud/core/api`), shared with every surface and the origin catalogue job. |
| `COVERAGE_ARTIFACT_URI`         | String  | _(unset)_               | Coverage artifact root for currencies and granularity; unset serves `[]` and `null`.  |
| `ORIGIN_CATALOGUE_URI`          | String  | _(Required)_            | Origin catalogue root; a missing or unloadable catalogue fails startup.             |

---

## Error Handling

Errors use RFC 9457 problem detail envelopes:

| HTTP Status | Problem Code (`code`) | Trigger Condition                                                                                                                                      | Pointer                                                                                     | Retryable |
| :---------- | :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ | :-------- |
| `400`       | `request_invalid`     | Malformed request body, text length exceeding 200 characters, batch size outside 1–25, or extra keys.                                                  | The offending body field, named by the platform's own request validation (e.g. `/texts/0`). | No        |
| `400`       | `request_invalid`     | `dcid` path parameter exceeding 512 characters.                                                                                                        | `/dcid`                                                                                     | No        |
| `400`       | `request_invalid`     | `POST /api/graph/nodes` with `dcids` outside 1–50 entries, a malformed or over-long dcid, or extra keys.                                              | The offending body field (e.g. `/dcids/0`).                                                 | No        |
| `404`       | `node_not_held`       | Requested DCID does not exist in the upstream DCP graph.                                                                                               | `/dcid`                                                                                     | No        |
| `400`       | `request_invalid`     | `GET /api/graph/documentation/{dcid}`'s `dcid` is held, but is neither a `Dataset` nor a `Provenance` node -- a `Source` dcid given directly included. | `/dcid`                                                                                     | No        |
| `503`       | `service_unavailable` | `GET /api/graph/documentation/{dcid}`'s own Dataset children read came back truncated (more than 256 children, or a surviving `nextToken`).            | None                                                                                        | No        |
| `404`       | `not_found`           | Unmatched route or empty DCID path parameter (`/api/graph/node/`).                                                                                     | None                                                                                        | No        |
| `502`       | `upstream_error`      | DCP read failure with a 4xx upstream response status.                                                                                                  | None                                                                                        | Yes       |
| `503`       | `service_unavailable` | Read pool semaphore saturation or upstream 5xx/network errors.                                                                                         | None                                                                                        | Yes       |
| `504`       | `timeout`             | Resolution, node lookup (single or batch), or source documentation read exceeding configured deadline.                                                 | None                                                                                        | Yes       |
| `500`       | `internal_error`      | Unhandled internal or executor failures.                                                                                                               | None                                                                                        | No        |

The 502, 503, and 504 refusals describe an infrastructure outcome with no request field at fault, and carry no pointer.
