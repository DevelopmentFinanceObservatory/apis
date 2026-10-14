A query can ask to leave something out ("excluding humanitarian aid", "except Kenya",
"not from the United States", "other than grants", "everything but humanitarian aid").
Copy each such phrase, with the word that leaves the thing out, into an `exclusions`
entry's `phrase`, and the thing left out into its `left_out`; keep the thing left out
out of `variables`, `entities` and `dropped_qualifiers`. A
negating word that is part of an indicator's or group's own name stays in that field and
makes no exclusion ("non-concessional loans", "non-DAC countries", "official development
assistance excluding debt relief"). A restriction ("only grants") or a bound on time ("no
later than 2015") leaves nothing out.
