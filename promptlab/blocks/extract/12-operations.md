`operations` records a request to do something with a measure beyond showing it. Test a
phrase by deleting it: if what remains still names the same measure, the phrase is an
operation on that measure. "top 5 donors for education aid" minus "top 5" leaves "donors
for education aid", so "top 5 donors" is a rank over origin on "education aid". "GDP
growth rate in Ghana" minus "growth rate" names a different measure, so there is no
operation. The test only finds operations. Words that name a measure's direction or
type, such as inward, outward, imports, exports, net, gross, disbursements or
commitments, belong to the indicator and stay in the variable, even when an entity's
flow_role repeats them. A share, rate, ratio or total that is part of
an indicator's own name is not an operation: "share of women in parliament", "renewable
energy share", "total population", "total fertility rate", "remittances as percentage of
GDP". Comparing named places, a series over time ("trends", "over time", "how has aid to
Ethiopia changed") and a date window are what search shows, and are not operations. The
verbs:
- rank: an order or a top or bottom N. "top 5 donors" runs over origin, "which countries
  received the most" over destination, "which sectors got the most aid" over category,
  "countries ranked by GDP" over place. "which African countries received the most" runs
  over destination beside the entity "Africa" with `members` true, flow_role
  "destination" and role_span "received".
- aggregate: a sum across a group of entities the query names or implies. "total aid to
  the Sahel countries" runs over destination, and "Sahel" has `members` true, flow_role
  "destination" and role_span "to".
These axes describe operations only. An entity's flow_role still comes from the query's
own wording alone, and its role_span copies that wording. A place the query only locates
("literacy rate in Mali", "population of Chad") has flow_role "unspecified" and no
role_span, and so do places named bare side by side ("ODA Chad Niger"), whichever of
them usually gives or receives.
- growth: a change between years the query names ("change in aid to Kenya since 2010").
  A change with no year named is a series over time, not growth.
- share_of: one flow of money as a share of another ("share of health in total aid to
  Kenya" has the variables "health aid" and "aid", and share_of on "health aid"). A share
  published as its own indicator is one variable with no operation ("share of renewable
  energy in total energy", "share of women in parliament").
- explain: an explanation or advice about the data ("why did aid to Ethiopia fall in
  2023" has the phrase "why did fall").
- forecast: a projection to a year after today's year ("aid to Ethiopia in 2030").
- report: an overview or write-up ("a comprehensive overview of health spending in
  Kenya").
A phrase recorded as an operation is not also listed in `dropped_qualifiers`. A forecast
takes its target year with it: "aid to Ethiopia in 2030" has the phrase "in 2030" and no
entry in `dates`. Growth and explain keep their years in `dates`, since those years
choose the data: "change in aid to Kenya since 2010" has a range from 2010.
