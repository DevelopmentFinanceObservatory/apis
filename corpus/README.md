# Eval corpus

The cases the Search eval sends to the engine, and what a correct answer to each looks like. `src/search/eval/cases.py` defines the format as one pydantic model, `Case`. `just corpus-schema` prints its JSON schema.

## Layout

Each `corpus/*.yaml` file is a YAML list of cases. Measure families live in `corpus/families/<source>.yaml`, each a mapping from family id to family. A family id is unique across files. `corpus/direction/` is a separate harness with its own format.

`just corpus-check` (`scripts/eval_corpus.py check`) loads every case with its families and fails on a schema error, a duplicate id or YAML key, an unknown family, a `why` over 200 characters, or a `known_failure` naming an unknown check. It also prints each case that cannot gate, with the reasons.

## A case

```yaml
- id: dq-crs-061
  query: gender equality aid to Afghanistan
  why: Gender marker is dropped; only purpose codes 15170 and 15180 answer, or the qualifier is refused.
  tags: { theme: gender, entity: country, direction: destination_only }
  known_failure: { issue: 1156, checks: [measure] }
  expect:
    entities: [{ dcid: country/AFG, role: destination }]
    window: { kind: all }
    definite:
      measure:
        family: crs.disbursement_gross
        bound:
          {
            crsTypeOfFlow: CRSTypeOfFlow10,
            crsPurpose: [CRSPurpose15170, CRSPurpose15180],
          }
    no_data: { reason: qualifier_not_bound, category: absent }
```

| Field                  | Meaning                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `id`                   | Lower-case letters, digits and hyphens; unique across all files.                                                                 |
| `query`                | The question text.                                                                                                               |
| `door`                 | `raw_text` (default), `parsed` with a `parsed` list of `search.models.ParsedPart`, or `dataset_resubmit` with a `resubmit` body. |
| `resubmit`             | `variable_text`, `shape`, `slots`, and optional `entity_dcids`, `origin_dcids` and `facets`.                                     |
| `include_observations` | Asks for the observation preview on the request.                                                                                 |
| `slice`                | `main` (default) or `holdout`. Holdout cases never gate.                                                                         |
| `activation`           | `live` (default) or `pending_data:<dataset>` for data not yet imported.                                                          |
| `why`                  | Required. One line of at most 200 characters stating, in the present tense, why the expected answer is right.                    |
| `reuse`                | Optional sentence-panel id.                                                                                                      |
| `tags`                 | A mapping of tag name to a value or list of values.                                                                              |
| `known_failure`        | `{issue, checks}`: checks that fail on the current engine, with the issue tracking the fault.                                    |
| `expect`               | What a correct answer looks like.                                                                                                |

## The `expect` block

`kind` is `resolution` (default), `catalogue` or `declined`, or a list of them. A declined answer carries `declined: {reason}`. A catalogue or declined answer may assert `scope_place`, where `null` asserts that the answer names no place.

The status keys `definite`, `candidates`, `no_data`, `unsupported` and `mixed` are the statuses a resolution may return. A resolution case needs at least one. Fields under a status key apply only when that status returns. The fields beside the status keys apply to every accepted status, and a status key may not repeat one of them. Write `definite: {}` to accept a status with nothing of its own to assert.

| Field                        | Where            | Meaning                                                                                                                             |
| ---------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `measure`                    | shared or status | `family` (one id or a list, any matching) or an inline `shape`; plus `bound`, `open` and `stat_vars`.                               |
| `entities`                   | shared or status | The exact set of `{dcid, role}`, role `origin`, `destination` or `observationAbout`; `expands: true` marks a group read as members. |
| `provenance`                 | shared or status | `{origin, dataset, bound}`: the Data Commons and provenance dcid, replacing the family's, and whether the identity binds it.        |
| `window`                     | shared or status | `{kind: all}`, `{kind: latest}`, or `{kind: range, start, end}` with either end open.                                               |
| `unit`                       | shared or status | `{currency, prices, base_year}` as the identity's unit request, or `none`.                                                          |
| `operations`                 | shared or status | `[{verb, over, target_year}]` in query order; `[]` asserts none.                                                                    |
| `warnings`                   | shared or status | Warning codes the answer must carry; extra warnings pass.                                                                           |
| `assumptions`                | shared or status | `[{kind, binding_ref, text}]` in clause order; `kind` is required, the others asserted when set; `[]` asserts none.                 |
| `skills`                     | `definite`       | `{present, ids}`.                                                                                                                   |
| `include_families`           | `candidates`     | Families the offered candidates must include.                                                                                       |
| `reason`, `category`, `gate` | `no_data`        | The refusal; `reason` may be a list of accepted reasons.                                                                            |
| `code`                       | `unsupported`    | The unsupported code.                                                                                                               |
| `parts`                      | top level        | One status-keyed block per part, in query order. Exclusive with top-level status keys and shared fields.                            |

A measure's `shape` keys are `population_type`, `measured_property`, `stat_type`, `measurement_qualifier` and `measurement_denominator`. An omitted key is not asserted and `null` asserts absence. `bound` maps a slot to a value, or to a list for a set binding. `open` lists slots that must stay unbound. `stat_vars` pins exact StatVars only where the StatVar is the answer.

A `provenance` names the Data Commons (`dfo`, `base`, `un_system`) and the provenance dcid the answer's facets carry; naming either replaces the family's pair. `bound: true` asserts that `definition.identity.facets.provenance` is a value binding, which an answer carries when the part names or excludes a source, since the identity binds the provenance that served, or when bind elects a shape only an opt-in dataset publishes, which names that table, and `bound: false` asserts that it is not. Written alone, `bound` keeps the family's pair.

`{gl: "..."}` in place of a value marks it unconfirmed: the text says what the grounding pass must read from the graph.

## Families

```yaml
crs.disbursement_gross:
  origin: dfo
  dataset: provenance/OECDCreditorReportingSystem
  shape:
    {
      population_type: FinancialFlow,
      measured_property: amount,
      stat_type: measuredValue,
      measurement_qualifier: Nominal,
      measurement_denominator: null,
    }
  fixed: { financialFlowType: Disbursement, nettingBasis: Gross }
```

A family fixes what every query of one kind resolves to: the Data Commons `origin`, the provenance dcid `dataset`, a partial `shape`, the `fixed` slots every member binds, and optional `stat_vars`. The loader expands each family a case names into `Measure.forms`, one form per family, merging the case's `bound` over the family's `fixed`; a case binding a fixed slot to another value is an error. A case with an inline shape gets one form with no provenance.

## Eligibility

A case gates when it sits in the `main` slice, its activation is `live`, it holds no `gl` value (its families included), and every check its `expect` asserts has an evaluator. `scripts/eval_corpus.py check` prints the reasons for every other case.

## Expectation hash

`search.eval.cases.expectation_hash` is the sha256 of the case's query, door, door payload, `include_observations` and `expect` as authored, with families expanded. Editing `why`, `tags`, `slice`, `activation`, `reuse` or `known_failure` leaves it unchanged. Editing a family changes it for every case that names the family. The paired gate treats a case whose hash differs from the base corpus as new.
