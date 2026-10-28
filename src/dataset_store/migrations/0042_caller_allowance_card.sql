-- Presentation card creates and revisions spend a caller allowance of their own, `card`
-- (`datasets_api.allowance`), kept beside the share and pin allowances.
--
-- The CHECK is replaced, and validating it reads the table under ACCESS EXCLUSIVE. The table
-- holds a row per caller, kind and day, so this is short, and `lock_timeout` fails it after five
-- seconds rather than queue every spend behind it. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.caller_allowance
    DROP CONSTRAINT IF EXISTS caller_allowance_kind_check,
    ADD CONSTRAINT caller_allowance_kind_check
        CHECK (kind IN ('share', 'card', 'pin', 'pin_budget', 'open_pin_budget'));
