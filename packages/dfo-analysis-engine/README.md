# dfo-analysis-engine

`dfo-analysis-engine` computes the nine analysis verbs (`aggregate`, `compare`, `filter`, `growth`, `join`, `quotient_of`, `rank`, `share_of`, `sum_of`) over `dfo_frame_grammar.Frame` objects. It has no store, no network and no web framework. `evaluate(recipe, input_frames, policy=...)` computes a recipe tree over its root frames and returns the result frame, its `ops_`/`ocv_` pair, the latest period its rolling op-specs count from, its exclusions, its execution identity and a JSON summary of the facts a caller renders. A declined call raises `AnalysisRefused`, whose `refusal` holds a stable code, retry mode, message and `fix`.

## Modules

| Module                                        | Responsibility                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `evaluate`                                    | `evaluate`: a recipe tree over its root frames, with one latest period per tree.                             |
| `calls`                                       | One verb call: the checks around the engine, the engine call, the result's ids and summary.                  |
| `anchor`                                      | The latest period of the root frames a rolling recipe reads.                                                  |
| `refusal`                                     | Refusal codes, retry modes and messages, and the one mapping from engine exceptions to them.                 |
| `engine`                                      | One `run_<verb>` function per verb, over Arrow tables, and the helpers they share.                            |
| `op_spec`                                     | The pydantic op-spec of each verb.                                                                            |
| `identity`, `execution_identity`              | The `ops_` op-spec hash, the `ocv_` content version, and the engine version they bind to.                     |
| `artifact`, `result`, `steps`, `units`        | The result, step and unit shapes a verb reports.                                                              |
| `rolling`, `recipe`                           | Rolling populations and the anchor key; a recipe's normal form, `normal_form_hash` and `replay_key`.          |
| `place_type`, `time_grain`, `unit_refs`, `price_basis`, `hashed_fields`, `frame_ref` | Small pure helpers the modules above share, and `FrameRef`. |

The tests under `tests/` pin every `ops_`, `ocv_` and op-spec hash against recorded vectors, and replay a recorded set of engine cases.
