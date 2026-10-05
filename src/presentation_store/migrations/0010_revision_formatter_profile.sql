-- Stores the formatter profile of the catalogue release that minted each revision. A read
-- formats the revision's fact summary from this column, so every stored revision reads under
-- any deployment, whatever release it pins.
--
-- The UPDATE fills rows minted under catalogue release 2026.8.0 with that release's
-- `formatter_profile`, copied from the `dfo_spec_kit_components` bundle manifest. A row
-- stamped with any other release keeps NULL, and SET NOT NULL then fails the migration, so
-- no row is ever given a profile its release did not carry.
--
-- The default is the same 2026.8.0 profile. It exists for the rows that a revision of the
-- service without this column inserts while a deploy rolls out; that revision mints every row
-- under 2026.8.0. `presentation_store.store` writes the column on every insert.

ALTER TABLE presentation.component_revision ADD COLUMN formatter_profile jsonb;

UPDATE presentation.component_revision
SET formatter_profile = '{"magnitude_words": {"billions": "billion", "millions": "million", "thousands": "thousand", "trillions": "trillion", "units": ""}, "period_display": {"day": "YYYY-MM-DD", "month": "YYYY-MM", "year": "YYYY"}, "significant_digits": 3}'::jsonb
WHERE catalog_version = '2026.8.0';

ALTER TABLE presentation.component_revision
    ALTER COLUMN formatter_profile SET DEFAULT '{"magnitude_words": {"billions": "billion", "millions": "million", "thousands": "thousand", "trillions": "trillion", "units": ""}, "period_display": {"day": "YYYY-MM-DD", "month": "YYYY-MM", "year": "YYYY"}, "significant_digits": 3}'::jsonb,
    ALTER COLUMN formatter_profile SET NOT NULL;
