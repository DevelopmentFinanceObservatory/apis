-- Drops the collection tables. Datasets registers and mints single definitions only, and no
-- code reads or writes these tables.
--
-- `version_record` rows with `subject_kind = 'collection'` stay. The Datasets routes accept
-- `spec_` ids alone, so a `col_` id never reaches them and refuses `not_found` at routing.
--
-- `version_member` references `version_record`, and `collection_member` references
-- `collection_registry`, so each referencing table drops first. `IF EXISTS` makes a repeat run
-- a no-op.

DROP TABLE IF EXISTS datasets.version_member;
DROP TABLE IF EXISTS datasets.collection_member;
DROP TABLE IF EXISTS datasets.collection_registry;
