# Facet election diff

Offline harness for #288. It runs the three facet elections an answer can take (the search pair plan, the search bulk plan and the mint) over the coverage view of a staged graph release and reports where they disagree. It needs no live service: everything reads the staged release and origin catalogue.

## Stage the data

```bash
mkdir -p scripts/facet_election_diff/data
uv run python scripts/stage_eval_data.py --checkout . --dest scripts/facet_election_diff/data \
  --release-uri <graph release URI> --catalogue-uri <origin catalogue URI>
```

The release lands in `data/graph-release` (about 240 MB for the dev release) and the catalogue in `data/origin-catalogue`. Set `FACET_DIFF_DATA` to use another directory. Every script reads from and writes to that directory.

## Run

Each script runs through `run.sh`, which puts this tree and this directory on `PYTHONPATH`, runs under `nice -n 10` and starts in the data directory. Load the app's settings into the environment first.

| Script | Writes | Measures |
| --- | --- | --- |
| `me_pops.py` | `populations.json` | The populations: the three largest index families, single-constraint sibling groups, multi-method StatVars with their top recipient and donor, and the eval corpus cases. |
| `me_m24.py [source] [window]` | `m24_results[_source][_window].jsonl` | Pair, bulk and mint elections per population and window, compared on lead, per-member facets, unserved members and continuation; and for groups, whether counting most-holders within the window moves the agreement key. |
| `me_summary.py` | stdout | Disagreement counts by source from the `m24_results_*.jsonl` files. |
| `me_compare.py` | stdout | Disagreement counts between two runs kept in `before/` and `after/`. |
| `me_classify.py` | stdout | Each disagreement by kind, with examples. |
| `me_m3.py` | `m3_results.json`, `m4_exhaustive.json` | The series and series-years "one method per member" drops, and every sibling group whose most-held key differs between lifetime and window counting. |
| `me_m1.py <0-2> <bulk\|probe> <window>` | `m1_results.jsonl` | Wall time and peak memory of today's bulk election against member probes plus one group election, on one of the three largest families. Run one stage per process. |
| `me_declared.py` | `declared.json` | The `defaultMeasurementMethod` each multi-method or ONE Climate Finance StatVar declares on the live graph. Needs graph credentials. |

```bash
scripts/facet_election_diff/run.sh me_pops.py
scripts/facet_election_diff/run.sh me_m24.py
scripts/facet_election_diff/run.sh me_m24.py multimethod 2022-2027
scripts/facet_election_diff/run.sh me_summary.py
```

The pair election comes from `read_plan._elect_groups`, the bulk election from `read_plan.role_election` over `role_scope.bulk_candidates`, and the mint from `_select_member_facets` for one member and `select_group_facets` for several, so each runs the production code of the tree it sits in. To compare two heads, run `me_m24.py` in each tree against the same data directory and move the results into `before/` and `after/`.

## Results on the dev release

On graph release `6d204cb9a34d` with #287 applied, 3,933 records in the 2019 to 2023, 1995 to 2000 and latest windows plus 790 in 2022 to 2027 show no pair or mint disagreement, and the bulk plan disagrees with the mint in 2 records: the climate finance sibling group `AttributedAmount_FinancialObligation_Commitment_*` in 2019 to 2023, whose Unspecified-objective member holds facet `12652164602495439961` with one point, in 2018. Today's bulk election takes 0.35 to 0.46 s on the largest CRS family (12,841 members, 1.04M coverage rows); member probes plus one group election take 3.4 to 3.8 s.
