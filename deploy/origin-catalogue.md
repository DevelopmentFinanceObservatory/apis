# Origin catalogue (`origin_catalogue`)

The origin catalogue lists every provenance the platform can serve, with the variables and topics each one holds and the Data Commons it comes from: DFO's own Spanner graph, the base Data Commons behind our mixer, and the UN System Data Commons. The apis service and the datasets worker load it at startup and answer origin questions from memory.

## Layout

`ORIGIN_CATALOGUE_URI` (`gs://` or `file://`) is the bucket root. Each image reads and writes only its own format root, `<ORIGIN_CATALOGUE_URI>/format-<N>/`, where `N` is `origin_catalogue.schema.CATALOGUE_FORMAT_VERSION` (currently 3, so `format-3/`). A format root holds:

- `current.json`, the pointer to the live build. The job writes it last, so it is the commit point.
- `versions/<build_id>/`, one directory per build: `provenances.json`, `dfo_variables.parquet`, `base_variables.parquet`, `provenance_places.parquet`, `withheld_variables.parquet`, `topics.parquet`, `scan.json`, `un.json` (the places the UN front door refuses to have named), `dataset_facts.json` (each source-index entry's card facts), `superseded_variables.parquet` (each superseded base variable and the UN variable standing for it), the five UN table files (`un_variables.parquet`, `un_arcs.parquet`, `un_excluded_variables.parquet`, `un_topics.parquet`, `un_mirror.json`) and `manifest.json`. A directory holding a manifest is complete.
- `lease.json`, the lease that keeps two runs from building at once.
- `catalogue_state.json`, the identity last seen unpublished and when it was first seen, which drives the stale-publish log event.

## What the job builds

`python -m origin_catalogue.job` builds these tables:

- The provenance table has one row per admitted provenance. DFO rows come from Spanner, base rows from `<-source` on every registered base source through our mixer (`typeOf: Provenance`, minus DFO's provenances and each entry's exclusions), and UN rows from the registry, each read from the UN deployment together with its agency's root group `undata/g/<segment>`. A DFO or base row carries its first `isPartOf` parent typed `Dataset` as `dataset` and takes the Dataset's name; a row with no such parent is an orphan. A UN row takes the registry label and has no `dataset`.
- The dataset facts hold the card facts of each DFO and base entry of `GET /api/graph/sources` (a Dataset, or an orphan provenance): description, licence label, raw release-cadence token, years covered and source URL. `origin_catalogue.card_facts` derives them from the arcs the provenance stage already reads and one batched read of the `name` of every `licenseType` node those arcs name, over the admitted provenances naming the Dataset in `isPartOf`, with the rules the documentation route applies, so a card equals its documentation page for the graph the build read. The provenance stage runs on every build, including one that carries the base tables forward, so the facts refresh at every identity change.
- The variable tables list DFO's variables, and the base variables with the admitted base provenances that hold each and the scan places where they have data.
- The provenance places list, for each admitted base provenance holding a base variable, the scan places where it holds one. A base variable's row unions its provenances and its places, so this table is what says which publisher has data in which country; the search catalogue counts each country's publishers from it. Pass 2 fills it from the facets it already reads.
- The withheld variable set lists each variable our mixer holds data for at a scan place under a provenance that is neither an admitted base provenance nor DFO's, with no facet under an admitted base provenance at any. A DFO facet neither admits a variable nor withholds it, so a variable whose facets are all DFO's is in neither table. Search refuses withheld variables as candidates. A variable in neither table passes as base, and every facet read still admits facet by facet. The job reads facets with no domain filter to tell a withheld variable from one with no data at all, which costs about 1.1 to 1.5 times the filtered read.
- The topic table lists the variables the topic walk reaches from each topic.
- The UN tables copy what search reads about a UN candidate between recalling it and checking its places, so deriving a shape for one sends the UN deployment nothing. See UN tables.
- The superseded variable table lists each withheld variable of a base provenance a UN agency supersedes, with the UN variable that stands for it where one does. See Superseded variables.

The job also probes which scan-set countries the UN front door refuses to have named in an observation GET, so the origin client leaves them out of UN reads. A country counts as refused only when the front door answers a GET naming it alone with its HTTP 403 page twice.

## UN tables

`origin_catalogue.un_mirror` reads every registered `[[un_system]]` agency from the UN deployment and writes five files:

| File | Rows | Answers |
| --- | --- | --- |
| `un_variables.parquet` | One per admitted variable (84,813): `variable`, its `series` (the `populationType` target) and `arcs`, row numbers into `un_arcs.parquet`. | The `->*` arcs of a UN variable, from which search reads the five-tuple, the constraint properties and values, `observationProperties`, the name and the type. The rows of one `series` are the members a UN series shape takes its slot keys from. |
| `un_arcs.parquet` | One per distinct arc target (112,188): `property`, the target's `dcid` or literal `value`, its `name` and its `types`. | The label of a constraint property, a constraint value and a series, each the `name` of an arc that names it. |
| `un_excluded_variables.parquet` | `variable` and `series`, every member variable of a series the registry excludes (39 variables of 37 series). | Admission of a UN candidate by whole dcid. These variables have no row in `un_variables.parquet`. |
| `un_topics.parquet` | `topic` and `leaves`, each topic under `undata/topic/Root` with the admitted variables the topic walk reaches from it (32,292 topics). | The expansion of a recalled UN topic. A variable of an excluded series, or of an agency the registry does not list, is no leaf. |
| `un_mirror.json` | `exclusions`, each excluded series and the DFO provenance its entry names, `series_names`, each excluded series and its name, and `agencies`, each agency's series, variable, excluded series and excluded variable counts. | The name of an excluded series, and the check of a carried table against the registry. |

UN topics have their own table because the topic table holds our mixer's topics and is copied forward under the base tables' key, while the UN tables follow the UN stage.

The stage finds an agency's variables by walking its StatVarGroup tree from `undata/g/<segment>`: `<-specializationOf` for each level's groups, `<-memberOf` for each group's variables. It then reads every variable's `->*` arcs, 25 variables a request. A series' own `<-populationType` list repeats its page token past 1,000 members, so the walk goes through groups, which are small. A read naming 25 groups that stops advancing is read again one group at a time.

The UN deployment sometimes answers a page request with the first page and a token the read has already been given, and answers the same request correctly a second later. The stage drops such an answer and requests the page again after 1, 2 and 4 seconds. A page that repeats a token on all four attempts stops the read, which is `un_enumeration_stalled`. Each repeated page adds at most 7 seconds, and a wait that would pass the stage's deadline is not taken. A node that repeats on every attempt costs its agency 14 seconds, 7 in the read naming 25 nodes and 7 in the read naming it alone, so 14 agencies and the topic tree stalling at once add under 4 minutes. `origin_catalogue_un_mirror_enumerated` reports the pages requested again as each agency's `page_retries` and the topic tree's `topic_page_retries`.

The table is complete or the build holds none of it. The stage reads the UN deployment's own count of the variables under each agency's root group and under each of the root's child groups, from the site's `POST /api/variable-group/info` (`descendentStatVarCount`), which the deployment computes from its group index in one unpaged response. The walk must find the same child groups and the same count under each. Every variable must name a series, and every excluded series must be the series of at least one variable and carry a name in that variable's `populationType` arc. The topic walk must find arcs for every topic and peer group it names, since every node has a type, and must reach at least one topic holding an admitted variable. A variable must sit under one agency's groups and carry that agency's id prefix, and the tables must hold together (`origin_catalogue.un_mirror.UnMirror`). A walk that falls short of any of these is an incomplete enumeration, which the build treats as a failed UN read (see UN carry-forward). An excluded series no variable names, or none names with a name, is a registry fault and refuses the build.

The stage keeps at most 4 requests in flight against the UN deployment. All 14 agencies come to 1,727 series and 84,852 variables in about 8,400 requests and 8 minutes, of which the topic tree takes 2,300 requests. The UN front door answers HTTP 429 to a client that has sent it tens of thousands of requests within the hour; the stage repeats such a read after 10, 30 and 60 seconds, then treats it as a failed UN read. The five files total 3.2 MB and decode to 67 MB of objects. The loader reads and converts each file 4,096 rows at a time, and a process that reads the UN tables alone grows by about 135 MB resident. The writer decodes, measures and releases the tables before it decodes the UN tables, so one decoded copy is alive at a time. A job run that carries the base tables and copies the UN tables (build `20261007T143810Z-4cd497d0`) peaks at 556 MiB resident. One that reads the published build, rebuilds the base tables from the mixer and copies the UN tables (build `20261007T155855Z-cf4a80b3`) peaks at 807 MiB, inside the job's 2 GiB.

The admitted sources come from `packages/dfo-data-tools/src/origin_catalogue/data/registry.toml`, which ships in the wheel. Adding or removing a source is a pull request to that file.

A base entry's `exclusions` name the provenances it holds that stay out. Each is `superseded_by_dfo`, an upstream copy of a dataset DFO imports, or `superseded_by_un_system`, which names a `successor`, a registered UN provenance whose series replace it. It also declares `variable_prefix`, `code_delimiter`, `pair_delimiter` and `value_delimiter`, from which the build derives each superseded variable's successor (see Superseded variables). `dc/base/UN_SDG` is superseded by `undata/p/SDG`. A base variable held only under an excluded provenance is in the withheld variable set, so it is a candidate, a topic member and a served StatVar on no door, with the UN origin on or off. A UN entry's `exclusions` name the series it does not admit, each `superseded_by_dfo` with the `dfo_provenance` that holds the data the series derives from. The registry lists 37 such series.

## Superseded variables

`superseded_variables.parquet` has one row per superseded variable: `variable` and `successor`, the UN variable that stands for it, null where none does. A superseded variable is a withheld variable of a base provenance a UN agency supersedes that DFO does not hold. `origin_catalogue.successors` derives the table on every build from the withheld variable set, the registry and the UN tables the build publishes, whether enumerated, copied or carried forward.

A `superseded_by_un_system` exclusion declares how a variable id of the superseded provenance reads. `variable_prefix` starts every such id. After it come the series code, then `code_delimiter` and the constraint pairs, joined by `pair_delimiter`, each a dimension and a value joined by `value_delimiter`. For `dc/base/UN_SDG` these are `sdg/`, `.`, `__` and `--`, so `sdg/AG_PRD_FIESMS.AGE--Y_GE15__SEX--F` reads as code `AG_PRD_FIESMS` with pairs `AGE=Y_GE15` and `SEX=F`. The successor agency's variable ids take the same form after `undata/<segment>/`. The table lists every withheld variable carrying a declared prefix that DFO does not hold; a variable DFO holds keeps its DFO data and has no row.

A row's successor is the agency's variable of the same code whose constraint pairs equal the superseded variable's as a set, order ignored, when the UN tables admit it. A row names no successor when that variable belongs to a series the registry excludes, and when the UN tables hold no variable of that code and those pairs: where the UN names a dimension or spells a value otherwise, or holds the breakdown not at all. Two agency variables of one code and one set of pairs refuse the build (`un_successor_ambiguous`), even when one of them has the superseded variable's own id. Every superseded variable stays withheld, with a successor or without one.

The dev build `20261007T155855Z-cf4a80b3` lists 4,635 `sdg/` variables, and 1,459 have a successor: 1,340 name the UN variable of the same id and 119 one holding the same pairs in another order. 31 name a variable of an excluded series, and 3,145 match no UN variable, mostly where the UN renamed a dimension or spells a value otherwise. `origin_catalogue_job_published` reports these counts as `successors` (`twins`, `reordered`, `series_excluded`, `unmatched`, `held_by_dfo`), and the rows as `row_counts.superseded_variables` and `row_counts.successor_variables`.

Every successor is an admitted variable of the same build's UN tables. The job refuses a row that names any other variable (`un_successor_not_held`), and the loader refuses a build that holds one or lacks the file.

Request code reads the table through two `Catalogue` methods. `successor(dcid)` returns the successor variable, and `None` for a variable with none and for every variable with the UN origin off. `is_superseded(dcid)` is true for every variable the table lists, with the UN origin on or off, and for a base Topic that the member cap admitted before supersession and that has no standing leaf since. A Topic the cap refuses for its size is not superseded.

The member cap counts every superseded leaf as the base leaf it was, so supersession admits no Topic the cap refused before it. A Topic within the cap is admitted when a leaf stands: an admitted leaf, or, with the UN origin on, a leaf with a successor, so search can expand the Topic and replace its leaves. A Topic admitted on such leaves alone is admitted as UN System. On the dev build 949 of 1,864 Topics have a superseded leaf and the cap admitted 1,533 before supersession. 1,165 are admitted with the UN origin on and 662 with it off, and 368 and 871 respectively are superseded.

## Serving the UN tables

The apis service and the datasets worker load the UN tables with the rest of the build (`origin_catalogue.artifact.load_catalogue`) and hold them in the snapshot as `CatalogueSnapshot.un`. The loader refuses a build whose UN tables break their invariants, hold a variable no agency's `undata/<segment>/` prefix starts, or disagree with the manifest's UN row counts or agency counts, and a failed reload keeps the last good snapshot serving.

Admission of a UN candidate reads the tables: `admit_candidate` and `data_commons` name the UN System for a variable `un_variables.parquet` holds and for a topic `un_topics.parquet` holds, and for no other `undata/` dcid. A variable of an excluded series, and a variable the UN added after the build, is no candidate, no topic member and no served StatVar.

`origin_catalogue.graph.OriginGraphClient` answers these reads of a UN dcid from the snapshot bound to the request and sends the UN deployment nothing:

| Read | Answered for | From |
| --- | --- | --- |
| `node_arcs`, `node_arcs_batch`, `node_arcs_many`, `node_arc_values`, `node_observation_properties_batch`, `node_names_many`, `exists`, `exists_batch` | A variable the tables hold | Its arc rows, without the `provenanceId` the UN deployment adds to each target |
| `node_arcs`, `node_arcs_batch`, `node_arcs_many`, `exists`, `exists_batch`, `node_type`, `node_types_batch` | A topic the tables hold | Its leaves as `relevantVariable` targets and the type `Topic`, so `topic_walk.walk_topics` reaches the same variables in one level |
| `node_label`, `node_labels_batch`, `node_type`, `node_types_batch` | A variable, and a node a variable's arc names: a constraint property, a constraint value, a series | The variable's `name` arc, or the `name` and `types` of the arc row naming the node |
| `node_subjects`, `node_subjects_many` | A series the tables hold, named as the `populationType` lead or as that filter's value | The series' admitted members that hold every other arc the read names, in sorted order, and a truncated page past `max_results`. A series the registry excludes has no subjects |

A variable of an excluded series reads as an absent node on each of them. A UN dcid the tables hold no fact for is read from the UN deployment: a unit, the arcs and alternate names of a constraint value or a series, the name of a topic, `enum_members`, and a reverse read that names no series. With the UN origin off the tables answer nothing and nothing is read.

Observation reads name only the UN variables the snapshot admits. Any other `undata/` variable reads as holding no data and costs no request, on Search and on the Datasets doors alike. A mint naming a variable of an excluded series refuses `member_not_resolvable`, since the build holds no node for it and the door cannot resolve the member. A mint naming a base variable whose only data sits under `dc/base/UN_SDG` refuses `excluded_by_registry`: the withheld re-read (`OriginGraphClient.withheld_at`) reads that variable from our mixer, where its data sits, and names every provenance holding it. Only `undata/` variables are re-read from the UN deployment. A UN variable the tables do not hold is not re-read, and its mint refuses `default_facet_undefined`.

Request code reads the tables through the `Catalogue` protocol: `un_variable(dcid)` for a variable's series and arcs, `un_series_members(series)` for every admitted member of a series, `un_topic_leaves(topic)`, `un_node(dcid)` for the name and types of a node an arc names, and `superseded_by(variable)` for the DFO provenance an excluded variable's series names.

The memory budget, `ORIGIN_CATALOGUE_MAX_RESIDENT_BYTES` (default 384 MiB), is checked against the manifest's `resident_bytes` plus `un_mirror.resident_bytes` before any table is read. The dev build records 115 MiB, 51 for the tables and 64 for the UN tables. Loading it takes 2 seconds and adds 247 to 266 MiB to a fresh process's resident set, about 2.3 times the recorded size. Two builds are resident during a swap. The dfo-dev apis service runs one process in a 4 GiB container, and the composed app with the Search and graph surfaces starts at about 1,170 MiB without the catalogue, so a build at the budget needs about 1,170 + 2 x 2.3 x 384 = 2,940 MiB, leaving about 1.1 GiB for the Datasets surface and a coverage index reload.

## Identity and skip

A build's identity hashes five inputs: the artifact format version, a hash of the job's modules (`origin_catalogue.job.JOB_MODULES`: the job, enumeration, the topic walk, the UN table enumeration, the successor derivation, the card-fact derivation, the artifact writer, the schema, the registry parser and the origin ids), the registry file's hash, the DFO fingerprint, and the ISO week. The DFO fingerprint is the latest successful `IngestionHistory` row plus the row count and latest `last_update_timestamp` over every `Observation` row. A registry edit, an ingestion, a new ISO week and an image that changes a job module each produce a new identity, so the base tables refresh weekly with no separate schedule and a job fix reaches the catalogue on its first run. A change to the serving modules (`catalogue`, `context`, `graph`, `service`) builds nothing.

Each run reads `current.json` and exits without building when it already carries the current identity (`origin_catalogue_job_noop`), unless that build carried its UN stage forward, in which case the run retries the UN (see UN carry-forward). It also exits without building while `IngestionLock` is held (`origin_catalogue_job_skipped`, reason `ingestion_lock_held`), because the fingerprint is unstable during a load. Before it exits, it checks that the identity the last unlocked run saw is published (see Monitoring).

## Lease

A run that needs to build takes the create-only `lease.json` at the format root. Cloud Scheduler starts an execution every 15 minutes and a full build takes longer, so most executions find the lease held and exit (`origin_catalogue_job_skipped`, reason `lease_held`). The lease lasts 7200 seconds, the job's task timeout, after which Cloud Run has killed the holder. An expired or unparseable lease is reclaimed with a write conditional on its version. The job releases the lease when it finishes.

## Carry-forward

The base variable, withheld variable and topic tables cost most of a build, since they read the mixer for every admitted base provenance. The job copies them from the published build when their key is unchanged: format version, a hash of the enumeration modules (`enumerate.py` and `topic_walk.py`), registry hash, the admitted base provenance set, DFO's provenance set, and the ISO week. DFO's variables and observations are not in the key, so an ingestion that adds no provenance publishes in minutes, new DFO variables included. A registry edit, a new base or DFO provenance, a change to the enumeration modules or a new week rebuilds the tables, within a deadline of 7200 seconds less a 1800-second reserve for the UN stage and a 900-second reserve for the topic walk and publication. The `origin_catalogue_job_published` log line reports `base_tables` as `carried` or `built`.

## UN carry-forward

The UN stage reads each registered agency's row and root group from the UN deployment, probes the front door, and reads the UN tables. It copies the UN tables from the published build when their key is unchanged (format version, a hash of `un_mirror.py` and `topic_walk.py`, registry hash, ISO week) and that build read them from the UN, so an ingestion sends the UN deployment only the row read and the probe. A registry edit, a change to those modules or a new week enumerates them again, within the job's 7200 seconds less the 900-second reserve. The `origin_catalogue_job_published` log line reports `un_mirror` as `enumerated`, `copied` or `carried`, and `origin_catalogue_un_mirror_enumerated` reports each agency's series, variables, the UN deployment's count, requests and seconds.

When a UN read fails or the enumeration is incomplete, the build carries the published build's UN rows, refused places and UN tables forward whole and publishes, so a UN outage delays no DFO or base update and no build holds a partial UN table. An enumeration is incomplete for one of eight reasons, each named in the carry-forward's `reason` with the dcid at fault:

| Kind | Meaning | What to do |
| --- | --- | --- |
| `un_enumeration_stalled` | A paged read of one UN node repeated a token on all four attempts at one page, or added nothing for three pages. | Read the node by hand. If the UN deployment still stalls, report it to the UN team. |
| `un_read_refused` | The UN front door refused a `/v2/node` read naming the node alone, with its HTML 403 page. A refused read naming several nodes is read again in halves until each read passes or names one node. The detail names the node and the property read. | Report the refused request body (`{"nodes": [<node>], "property": <property>}`) to the UN deployment's operators; the next run reads the UN again. |
| `un_variable_count_mismatch` | The group walk and the UN deployment's own count differ for an agency's root group or one of its child groups. The detail names up to five groups with both counts. | Wait for the next run. A difference that persists means a truncated list or a count the deployment has not refreshed; compare the named group by hand. |
| `un_topic_unanswered` | A topic or peer group the topic walk names, `undata/topic/Root` included, answered no arcs. | Read the node by hand; the next run reads the UN again. |
| `un_topics_empty` | The topic walk from `undata/topic/Root` reached no topic holding an admitted variable. | Read the root by hand; the next run reads the UN again. |
| `un_tables_inconsistent` | The tables break an invariant of `UnMirror`, such as an agency's variable naming another agency's excluded series. The detail names the variable or the invariant. | Report the variable to the UN team; the next run reads the UN again. |
| `un_variable_outside_agency` | An agency's group lists a variable whose id does not start with `undata/<segment>/` of that agency. The detail names up to three. | Report the variable to the UN team; the next run reads the UN again. |
| `un_variable_without_series` | A UN variable has no `populationType` target, so no series decides its admission. | Report the variable to the UN team. |

No registry entry decides whether a read advances or the front door admits it, a count agrees, a variable names its series, a topic answers its arcs or the tables hold together, so these are faults of the UN deployment's answer and delay nothing else. A UN row the checks reject (`un_agency_root_group_missing`, `provenance_without_url`) an excluded series no variable names (`un_excluded_series_missing`) and one no variable names with a name (`un_excluded_series_unnamed`) are refusals like any other, so a bad registry edit never publishes. A run that refuses on `un_excluded_series_missing` lists the incomplete agencies beside it. Carried rows keep their own segment and take the current registry's label. A carried row whose segment differs from the registry's is refused (`un_carried_segment_mismatch`), and an agency the published build does not hold stays out until the UN answers. Carried UN tables holding an agency segment the registry does not list, or lists under another provenance, are refused (`un_carried_agency_mismatch`), since admission of a UN candidate reads the tables. Carried UN tables whose excluded series differ from the registry's are refused (`un_carried_exclusions_mismatch`), since their variables were admitted under other exclusions. A base provenance superseded by a UN agency the build has no row for is refused (`un_successor_not_served`). The run logs `origin_catalogue_job_un_carried_forward` at `WARNING` with the reason and `from_build`, the build that last read the UN, and the manifest records the same as `un_carried_forward`. `origin_catalogue.verify` prints it. With no published build to carry from, a failed UN read or an incomplete enumeration fails the build.

A published build that carried its UN stage forward does not satisfy the identity skip, and its UN tables are never copied. The next run with the same identity retries the UN. If the UN answers, it publishes a build that read it. If the read fails or the enumeration is incomplete again, it logs `origin_catalogue_job_un_retry_failed` at `WARNING` and publishes nothing, since the published build already carries the same UN stage.

## Publication

A run discards its build (`origin_catalogue_job_discarded`) when the ingestion lock was taken or the fingerprint moved while it ran; the next run starts again. Otherwise it writes `versions/<build_id>/`, then writes `current.json` conditional on the generation it read at the start. A run that loses that race logs `origin_catalogue_job_lost_publish_race` and leaves an unreferenced build for pruning. Every run prunes builds that nothing can still be reading: a build is deleted only when `current.json` does not name it and both the pointer and the build are older than 7200 seconds.

## Monitoring

Every run that exits 0 logs `origin_catalogue_job_heartbeat` as its last line, with the published build's `build_id`, `built_at` and `age_seconds`. All three are `null` when nothing is published, and `age_seconds` is `null`, at `WARNING` with an `error`, when the published build's manifest is unreadable. A refused or failed run logs no heartbeat and exits 1.

`origin_catalogue_publish_stale` carries a `reason`:

- `identity_unpublished`: the current identity has gone unpublished for more than one cycle (`ORIGIN_CATALOGUE_CYCLE_SECONDS`). A run that finds `IngestionLock` held cannot compute its own identity, so it checks the identity the last unlocked run recorded in `catalogue_state.json`. A lock held past a cycle while that identity is unpublished logs the line on every run.
- `published_build_age`: the published build is older than a week plus 7200 seconds plus one cycle. The ISO week in the identity makes a healthy job publish at least weekly, and the first run of a week may wait a cycle for the lease and then build for up to 7200 seconds, so an older build means nothing has published since the week turned. The line comes just before the heartbeat, with `build_id`, `age_seconds` and `limit_seconds`.

dfo-infra's `origin_catalogue.tf` alerts on three conditions in one policy: a stale line, a failed execution, and no heartbeat for 9000 seconds. The last catches a Scheduler that stopped starting the job, which logs nothing and fails nothing. Its window exceeds the 7200-second task timeout plus one cycle.

## Refusals

The job refuses to publish a version when any check below fails. A refusal is whole-version: no partial catalogue is written, and the last good build keeps serving. The run logs `origin_catalogue_job_refused` with each refusal's `kind`, `subject` (the dcid at fault) and `detail`, then exits 1, so the Cloud Run execution fails, the failed-execution alert fires, and a deploy waiting on the job stops. The next run sees the identity still unpublished and, after one 15-minute cycle, logs `origin_catalogue_publish_stale` (reason `identity_unpublished`), which the stale-publish alert watches. A run that skips on an unchanged identity, the ingestion lock or a held lease exits 0.

| Kind | Meaning | What to do |
| --- | --- | --- |
| `dfo_provenance_under_unreviewed_source` | A DFO provenance sits under a registered base source whose entry states no `exclusions`. | Review the overlap and add `exclusions` to the entry in `registry.toml`, an empty list if none apply. |
| `dfo_provenance_without_source` | A DFO provenance has no `source` arc in the graph. | Fix the source arc in the ingestion that produced the provenance. |
| `provenance_without_url` | An admitted provenance has no `url` arc, or its URL has no hostname, and the domain filter matches facets by hostname. | Correct the `url` arc upstream, or exclude the provenance in the registry. |
| `exclusion_not_held` | A registry entry excludes a provenance its source does not hold. | Correct the dcid in the entry, or remove the exclusion after an upstream rename. |
| `base_source_without_variables` | A registered base source has admitted provenances and none holds an admitted variable. | Check the entry's `country` and `scan_type` and the source in the mixer. |
| `un_agency_root_group_missing` | A UN agency's root group `undata/g/<segment>` is absent on the UN deployment. | Check the entry's `segment`, then whether the UN deployment renamed the agency's tree. |
| `un_carried_segment_mismatch` | A UN read failed or the enumeration was incomplete, and a carried UN row's segment differs from the registry entry's. | Wait for the UN to answer, so the new segment's root group is checked, or revert the segment edit. |
| `un_excluded_series_missing` | A registry entry excludes a series that no variable of its agency names. | Correct the series id in the entry, or remove the exclusion after an upstream rename. |
| `un_excluded_series_unnamed` | A registry entry excludes a series that no variable of its agency names with a name in its `populationType` arc. | Correct the series id in the entry, or remove the exclusion after an upstream rename. |
| `un_exclusion_provenance_not_held` | A registry entry excludes a series as superseded by a `dfo_provenance` with no observations in Spanner. | Correct the provenance in the entry, or ingest the data first. |
| `un_carried_agency_mismatch` | A UN read failed or the enumeration was incomplete, and the carried UN tables hold an agency segment the registry no longer lists, or lists under another provenance. | Wait for the UN to answer, so the tables are enumerated under the current registry, or revert the removal. |
| `un_successor_ambiguous` | Two variables of the successor agency hold a superseded variable's series code and constraint pairs. | Check the supersession's delimiters against the agency's variable ids. |
| `un_carried_exclusions_mismatch` | A UN read failed or the enumeration was incomplete, and the carried UN tables' excluded series differ from the registry's. | Wait for the UN to answer, so the tables are enumerated under the new exclusions, or revert the exclusion edit. |
| `un_successor_not_held` | A successor row names a variable the build's UN tables do not admit. | Check the `variable_prefix` and `code_delimiter` of the supersession against the successor agency's variable ids, then `origin_catalogue.successors`. |
| `un_successor_not_served` | A base provenance is excluded as superseded by a UN agency the build has no row for. | Wait for the UN to answer, or revert the supersession. |

A run also exits 1 (`origin_catalogue_job_failed`) for an input the catalogue cannot represent: no successful ingestion to fingerprint, an observation whose facet has no `TimeSeries` provenance, a source whose `<-source` list exceeds 5000 subjects, a malformed registry, a base-table build that runs out of time, tables that break the catalogue's invariants, a failed read of our mixer, or a failed UN read or an incomplete enumeration with no published build to carry forward. The Cloud Run execution fails, and the failed-execution alert fires.

## Registry pull request before ingestion

A registry pull request precedes an ingestion that adds a DFO provenance under a registered base source. An ingestion moves the fingerprint, so the next run builds with the new provenance. If its source entry has no reviewed `exclusions`, the run refuses, and no new build publishes for any reason until the registry is fixed. Merging the registry change first, with `exclusions` stating which overlapping base provenances stay out, lets the ingestion publish on its first run.

## Stale catalogue

`CatalogueSnapshot.is_stale` reports a loaded build as stale once its `built_at` is more than 14 days old (`origin_catalogue.catalogue.STALE_AFTER`). The weekly identity change means a healthy job publishes at least every 7 days, so a stale catalogue means the job has been failing or refusing for a week or more. Read the job's logs for `origin_catalogue_job_refused` or `origin_catalogue_job_failed`.

## First deploy and format bumps

The first deploy has no published build, and a format bump reads a new, empty format root, so neither has a build to carry forward or serve. `origin_catalogue.verify` fails on such a root, so `deploy-dev.yml` executes the job with `--wait` and verifies again before either service moves (see Deploy order). With nothing to carry forward, the build is a full one, UN tables included, and takes up to two hours. A build published from a local run shortens the wait: publish it to the bucket (below), and the deploy's first verify passes.

A format bump starts a new format root and leaves the previous root's last build in place, so a rollback to the previous image still finds its catalogue.

An image reads only its own format root. Where the bucket holds a build under another format root and none under the image's own, that image's first deploy runs the full build before either service moves.

## Running it locally

`--local-dir` publishes under a local directory in place of `ORIGIN_CATALOGUE_URI`. The run needs credentials for BigQuery and network access to the mixer, the UN deployment's API and the UN site:

```bash
export ORIGIN_CATALOGUE_BQ_CONNECTION=dev-fin-observatory.northamerica-northeast1.dfo_dev_dc_spanner_connection
export DATASETS_GRAPH_BASE_URL=https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app/core/api
uv run python -m origin_catalogue.job --local-dir /tmp/origin-catalogue-local
```

`--local-dir` replaces `ORIGIN_CATALOGUE_URI`, which need not be set. Point a local service's `ORIGIN_CATALOGUE_URI` at `file:///tmp/origin-catalogue-local` to serve against the build.

To seed the bucket, copy the format root: `gcloud storage cp -r /tmp/origin-catalogue-local/format-<N> gs://dfo-dev-origin-catalogue-<project>/`. Copy `versions/` before `current.json`.

## Environment

- `ORIGIN_CATALOGUE_URI`: required, `gs://` or `file://`.
- `ORIGIN_CATALOGUE_BQ_CONNECTION`: required, `project.location.connection_id`.
- `DATASETS_GRAPH_BASE_URL`: required, our mixer's API root including `/core/api`.
- `UN_GRAPH_BASE_URL`: optional, the UN deployment's API root including `/core/api`, defaulting to `https://unsd-datacommons.gcp.un-icc.cloud/core/api`. The apis service and the datasets worker read the same variable with the same default, since a build records UN rows, refused places and UN tables from the deployment it read.
- `UN_SITE_BASE_URL`: optional, the UN site root that serves the group counts, defaulting to `UN_GRAPH_BASE_URL` without its `/core/api`. A `UN_GRAPH_BASE_URL` that does not end in `/core/api` needs it set.
- `DATASETS_GRAPH_TOKEN`: optional bearer token, unset on the dev graph, which serves reads anonymously.
- `ORIGIN_CATALOGUE_BQ_PROJECT`: optional billing project, defaulting to the connection's leading segment.
- `ORIGIN_CATALOGUE_GRAPH_TIMEOUT_SECONDS`: optional, default 120.
- `ORIGIN_CATALOGUE_CYCLE_SECONDS`: optional, default 900, matching Cloud Scheduler's cadence.

## Deploy order

`deploy.yml` points the origin catalogue job (`dfo-dev-origin-catalogue` in dev) at the new image after the coverage steps and before the datasets worker and the apis service move, so the scheduled runs (Cloud Scheduler, every 15 minutes) use the new code. The services load the catalogue at startup and fail readiness when none can be loaded, so the deploy checks that a compatible build is published and leaves keeping it fresh to the scheduled job. A running service reloads a newer build within `ORIGIN_CATALOGUE_RECHECK_SECONDS`.

The check is `python -m origin_catalogue.verify`, run as an execution of the same job with its args overridden (the command stays `python`), so it reads the job's own `ORIGIN_CATALOGUE_URI`. It loads the new image's format root with the loader the services call at startup, reads the build's UN tables, prints both sets of row counts and each agency's counts, and exits non-zero unless a compatible build and its UN tables load. A build in another format's root does not satisfy it.

Verify runs first. It passes when the format root holds a compatible build, however old, and the deploy goes on without a build, so a refusal or a failed read in the job does not block a code release. The first deploy and a format bump have no build in the new format root, so verify fails; the workflow then executes the job with `--wait` and verifies again, and the deploy stops with both services on their current image if the build or the second verify fails. The second verify also catches a run that exits 0 without publishing, such as one that finds the lease held by a scheduled run still building.
