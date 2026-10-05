also_plausible (optional): a bare quantity name is often compatible with more than one real
reading of the SAME base concept — real ambiguity, not cosine ties. A genuine alternate is
either
  (a) a different interpretation of the query's METRIC itself (for "GDP", nominal vs.
      PPP-adjusted answer materially different questions about "GDP"); or
  (b) a DENOMINATOR variant of the identical measured concept — same populationType, same
      base measuredProperty/metric concept, differing only in measurementDenominator (an
      absolute quantity vs. its per-capita, %-of-GDP, or share-of-total variant; for "health
      spending" all three qualify, each re-expressing the identical quantity). A normalized
      sibling may carry FEWER constraint slots than the absolute shape; that alone does not
      disqualify it. Rule (b) fires only when the query does not already pin the denominator:
      if the query is already denominator-framed ("per capita", "as a share of GDP", "% of
      total"), the shape matching that framing is what was asked for and
      different-denominator siblings are not alternates under (b), though they may still
      qualify under (a).
