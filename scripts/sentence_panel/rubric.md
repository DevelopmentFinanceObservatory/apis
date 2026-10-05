# Sentence assessment rubric

This rubric scores a run of the sentence panel (`just sentence-panel`), which sends every panel query to the deployed Search resolver. For each query in the run's digest, judge the sentence the user reads (the envelope `rendered_sentence`, and for candidates each candidate's sentence) as a user who typed that query. The user is an analyst, journalist or campaigner who knows the topic and does not know our data model.

Three questions per query:

1. Faithfulness. Does the sentence say what the engine will return, and does it account for every part of what the user asked? A request element the engine ignored without saying so is the worst failure.
2. Clarity. Could the user read it once and know what number they are about to see?
3. Sentence versus options. Sort every fact the response carries (bound slots, defaults, entities, window, source, unit, counts, `slot_options`, `alternative_measures`, warnings) into one of:
   - `sentence`: the user cannot interpret the number without it (for example, gross vs net when they differ a lot, the place, the year actually shown, a substitution of what they asked for).
   - `option`: a choice the user may want to narrow or switch to (sector breakdown, instrument, commitment vs disbursement, another donor).
   - `nowhere`: internal or redundant (observation counts, series counts, code-list jargon).

## Sentence types

Tag each sentence with one structural type:

| code         | shape                                                                           |
| ------------ | ------------------------------------------------------------------------------- |
| `D_place`    | definite, one indicator about a place ("X in Ghana")                            |
| `D_flow`     | definite, provider and recipient roles ("X to Kenya from Germany")              |
| `D_open`     | definite with an open role or place read from coverage ("across 78 recipients") |
| `D_group`    | definite where a group or region was named or expanded                          |
| `D_global`   | definite with no place or role                                                  |
| `C_header`   | candidates count and difference sentence                                        |
| `C_item`     | one candidate's sentence (assess the set together under the query)              |
| `N_variable` | no data: indicator not matched                                                  |
| `N_entity`   | no data: place or role not resolved, ambiguous or not admitted                  |
| `N_coverage` | no data: matched but nothing published, including near-match leads              |
| `U`          | unsupported                                                                     |
| `M`          | multi-part answer                                                               |
| `X`          | catalogue or declined                                                           |

## Issue codes

Tag every issue present, using these codes only:

| code                   | meaning                                                                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DROP_SILENT`          | a request element (time, operation such as rank or compare, price basis, unit, qualifier, breakdown, entity) is ignored and the sentence does not say so |
| `DROP_SIGNALLED`       | dropped, and the sentence says so                                                                                                                        |
| `SUBSTITUTE_SILENT`    | answers a different indicator, source or entity than asked without saying so                                                                             |
| `MISRESOLVED`          | an entity, role or indicator resolved to the wrong thing                                                                                                 |
| `JARGON`               | internal or code-list vocabulary the user would not use ("standard grant", "gross of fixed capital consumption", "financial position type")              |
| `STACKED`              | qualifiers chained so the sentence is hard to parse ("for X for Y for Z")                                                                                |
| `LIST_BLOWUP`          | a group expanded into a long list in the sentence                                                                                                        |
| `COUNT_NOISE`          | observation, series or entity counts that do not help                                                                                                    |
| `GRAMMAR`              | plural, article, capitalisation or ordering errors ("1 donors")                                                                                          |
| `ROLE_WORDING`         | the role preposition reads wrong for the data ("debt to Ghana", "from" where the user means "owed to")                                                   |
| `REDUNDANT`            | the same fact twice (unit twice, two different recipient counts)                                                                                         |
| `CONTRADICTION`        | the sentence disagrees with itself, the status or the headline                                                                                           |
| `DEFAULT_HIDDEN`       | a default that changes the number materially is not stated                                                                                               |
| `DEFAULT_LOUD`         | a default stated with more weight than it deserves                                                                                                       |
| `CANDIDATES_ILLEGIBLE` | the user cannot tell the candidates apart or see why they differ                                                                                         |
| `NO_DATA_DEAD_END`     | a no-data sentence gives no usable next step                                                                                                             |
| `GOOD`                 | no issue; the sentence is clear and faithful                                                                                                             |

Severity per query: 0 fine, 1 awkward, 2 confusing or partly unfaithful, 3 misleading (the user would take away a wrong belief about the number).
