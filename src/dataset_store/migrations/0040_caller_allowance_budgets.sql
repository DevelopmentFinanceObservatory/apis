-- The pin door's shared daily budgets join the caller allowances (`datasets_api.allowance`).
--
-- `pin_budget` counts the new pinned copies the pin door makes in a UTC day, and
-- `open_pin_budget` those it makes for callers without the pin grant. Each is one row a day
-- under the caller key `all`, spent by the same conditional upsert as a caller's allowance, in
-- one transaction with the caller's `pin` unit, so pins racing at a limit cannot pass it.
--
-- The CHECK is replaced, and validating it reads the table under ACCESS EXCLUSIVE. The table
-- holds a row per caller, kind and day, so this is short, and `lock_timeout` fails it after five
-- seconds rather than queue every spend behind it. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.caller_allowance
    DROP CONSTRAINT IF EXISTS caller_allowance_kind_check,
    ADD CONSTRAINT caller_allowance_kind_check
        CHECK (kind IN ('share', 'pin', 'pin_budget', 'open_pin_budget'));
