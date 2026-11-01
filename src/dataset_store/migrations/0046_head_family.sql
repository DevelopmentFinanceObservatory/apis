-- The member family a subject head last admitted, and the family a completed async operation's
-- run served.
--
-- The family is a fact of an admission, where the version record (`0004_version_records.sql`)
-- keeps the trace of the first mint of its bytes: bytes depend only on the members a run serves,
-- so a family that moves while those stay equal makes the same `content_version`.
--
-- `subject_head.family` is one JSON object (`materialise.trace.recorded_family_json`): the trace's
-- `family_member_digest` and `family_member_count`, both null when the admission could not state
-- the whole family, and `stub_candidates`, absent when there are none
-- (`materialise.trace.family_fields`); `confirmed_at`, the head confirmation it was stated at; and
-- `basis`, the member-resolution inputs it was resolved under (`adapter.family_basis`: the code
-- index content hash, the bound graph release id, the bound origin catalogue build id and the
-- materialiser image build). `LiveHeadStore.advance_head` writes the run's family and
-- `confirm_head` the family the freshness check resolved, each at the confirmation it writes and
-- under the serving basis. A head serve states the family in place of the version record's only
-- while `confirmed_at` equals the head's own and `basis` equals the serving one, and states it
-- unknown otherwise: on a head NULL here because no write has reached it since this migration, a
-- head an older image confirmed or advanced without writing the column, and a head whose family
-- another code index, graph release, catalogue build or image resolved. The family decides no
-- freshness and no rebuild. Such a head states unknown until its next recheck or run.
--
-- `operation.family` is the same JSON object: the trace fields the run that completed the row
-- served, the `confirmed_at` it completed with and the basis it resolved them under. A `done`
-- poll states them only while that equals the row's `confirmed_at`
-- (`0032_operation_confirmed_at.sql`) and the basis the serving one, and states the family
-- unknown otherwise. `LiveOperationStore.complete` sets it and a resubmit's reset
-- clears it, so it is NULL on a row completed before this migration and on one this image reset
-- since. An older image's reset leaves the column as it was. When that image completes the row
-- again at another confirmation, the poll states the family unknown. When its completion
-- writes the confirmation the stored family was recorded at, the poll states that family, the one
-- the head was confirmed with at that instant, while the basis holds.
--
-- Apply this migration before deploying an API or worker image that reads the columns. Every
-- `LiveHeadStore` statement in such an image names them, as does every `LiveOperationStore`
-- statement that returns an operation row, so against a table without them each fails with an
-- undefined-column error, surfaced as `DatasetsStoreUnavailable`. The abandoned-row sweep
-- (`fail_abandoned_operations`) and the stale-row delete (`delete_terminal_operations`) name
-- neither. An image older than the columns runs unchanged against the migrated tables and leaves
-- them as they are.
ALTER TABLE datasets.subject_head
    ADD COLUMN family jsonb;

ALTER TABLE datasets.operation
    ADD COLUMN family jsonb;
