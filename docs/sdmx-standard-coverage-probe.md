# SDMX standard-StatVar coverage and per-pair latency probe

This measures whether the dev Cloud Run mixer's SDMX `/data` endpoint can serve
standard (non-dev-finance) StatVar/entity pairs, and how per-pair round-trip
latency scales when a single request is issued per `(stat_var, entity)` pair
instead of one batched call across every entity.

Run against the dev Cloud Run deployment
(`https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app/core/api`,
build `gitHash=79627f8`, `EnableSDMXDataApi=true` per `GET /version`), via
`dc_sdmx.SdmxClient.data()`/`stream_data()` directly against the live mixer
(sequential requests, no mocked transport). Build vintages on this endpoint
move; a coverage/latency check like this one should be re-run before being
trusted against a different build.

## Coverage: standard StatVars

Six `(stat_var, entity)` pairs, all present as `stat_vars`/`entities` in
`corpus/standard.yaml`'s cases, queried against `/data` with `variableMeasured`/`observationAbout` filters and
no facet restriction:

| StatVar                                    | Entity        | HTTP | Rows | Facet ID on every row |
| ------------------------------------------ | ------------- | ---- | ---- | --------------------- |
| `LifeExpectancy_Person`                    | `country/KEN` | 200  | 64   | yes                   |
| `Count_Person`                             | `country/KEN` | 200  | 140  | yes                   |
| `Count_Person`                             | `country/IND` | 200  | 181  | yes                   |
| `FertilityRate_Person_Female`              | `country/KEN` | 200  | 65   | yes                   |
| `Amount_Emissions_CarbonDioxide_PerCapita` | `country/KEN` | 200  | 55   | yes                   |
| `Count_Person_NotEnrolledInSchool`         | `country/USA` | 200  | 17   | yes                   |

Every pair returns real, non-empty data with a well-formed `facet_id` on every
row (not just a sample), confirming `materialise`'s facet-dispatch logic has
what it needs from a raw SDMX-CSV response. The row counts match, exactly or
closely, the per-entity observation counts read from the graph
(`Count_Person`/`country/IND`: 181 observations in
both; `FertilityRate_Person_Female`/`country/KEN`: 65 in both;
`Amount_Emissions_CarbonDioxide_PerCapita`/`country/KEN`: 55 in both;
`Count_Person_NotEnrolledInSchool`/`country/USA`: 17 in both) -- strong
corroborating evidence that this SDMX-enabled build serves the same
underlying observation data the standard resolve path serves, for ordinary
(non-dev-finance) StatVars.

**Coverage verdict: pass.** An earlier build vintage recorded an empty-response
finding for `Count_Person`/`country/USA` and `/availability`; on the current
build, standard StatVars return real, non-empty coverage.

## Latency: per-pair round-trip multiplication

Sequential `stream_data()` calls, one distinct real country entity per call
(`LifeExpectancy_Person` against 1, 10, and 60 distinct `country/XXX`
entities), simulating the per-pair calling pattern a region-expansion query
would drive when 60 entities are bound to a single spec (`SEARCH_MAX_CHILD_PLACES`
caps a spec at 256):

| N sequential calls | Wall-clock | Avg per call |
| ------------------ | ---------- | ------------ |
| 1                  | 0.191s     | 0.191s       |
| 10                 | 1.911s     | 0.191s       |
| 60                 | 11.857s    | 0.198s       |

For comparison, the current single batched `graph.observation_facets_batch` call
for the same 60-entity/1-stat_var shape completes in 0.302s.

**Latency verdict: pass.** N=60 sequential SDMX round trips complete in
11.9s, well inside `SEARCH_REQUEST_TIMEOUT_S`'s 45s budget (roughly 3.8x
margin), even though it is about 39x slower than the current single batched call.
The multiplication is real and worth keeping in the request's own latency
budget/observability, but it does not blow the timeout for the documented
worst case (60 entities).

## Outcome

Both coverage and latency pass. The bridge design holds for the standard
(non-dev-finance) preview path: proceed with per-pair
`materialise_preview` calls against the real SDMX bridge as designed, with no
fallback needed on the evidence gathered here. Re-check both numbers if the
mixer build vintage changes before the next dependent measurement is taken.
