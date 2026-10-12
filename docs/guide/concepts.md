---
description: Understand how questions, specs, versions, StatVars, series, rows, and facets fit together.
---

# How the APIs fit together

The DFO APIs separate interpretation from data delivery. Search turns a question into a spec. Datasets materialises that spec into a versioned set of files.

## Check the interpretation first

Search returns a confirmation sentence with every definite spec. Read it before using the data. A useful sentence names the measure, donor and recipient, dates, source, and unit. `definite` means the API selected one interpretation. It does not establish that the interpretation matches the question you meant to ask.

For the question “ODA disbursements from Germany to Nigeria for basic health care”, a confirmation sentence identifies Germany as the donor, Nigeria as the recipient, and basic health care as the purpose. It also exposes decisions about gross versus net flows, the available time range, currency, and source.

## Specs and versions answer different questions

A spec records the canonical meaning of a question. Its `spec_id` is derived from that canonical definition. The definition includes the selected entities and bound or open dimensions.

A `content_version` identifies a stored materialisation of a spec. Keep the pair when citing or retrieving a result. Repeating a mint can reuse a version when the resolved definition, selected facets, source revision, and data are unchanged. A later materialisation can produce a new version when those inputs differ.

`GET /api/datasets/{spec_id}/{content_version}` retrieves a stored version while its artifacts remain servable. `GET /api/datasets/{spec_id}` serves a recently confirmed stored head or rechecks the definition and can materialise a newer version.

## StatVars, series, and rows

A StatVar identifies a measure with its dimensions. A question can resolve to more than one StatVar when it leaves dimensions open. “Basic health care” can cover several cooperation modalities or finance types, for example.

A series is one StatVar for one set of role entities and one selected facet. Each series has observations over time. A row is one observation in the downloaded table. Rows from separate series can share a year while representing distinct combinations of dimensions.

Do not add rows together without checking their dimensions and summability. A table can contain both aggregate and detailed categories, or rows from dimensions that overlap.

## Sources, units, and facets

A facet identifies the source, measurement method, and unit used for a series. The Search response and Datasets manifest record the selected facet because values with the same StatVar and entities can differ by source or unit.

The Datasets API follows a graph-declared default where one is available. Otherwise it applies a deployment fallback ranking. The fallback ranks candidates by observation count, latest observation, provenance, measurement method, unit, and facet identifier. When observation counts are unavailable, it uses the latest observation. A unit conversion request can select a native-currency input before conversion.

Bind `scope.facets` when a question requires a particular source, method, or unit. Provenance and measurement method use a `ref`. A unit can use a `ref` or a conversion request.

This fragment is the `scope` object inside a Datasets request's `input` object.

```json
{
  "scope": {
    "facets": {
      "provenance": {"ref": "provenance/OECDCreditorReportingSystem"},
      "unit": {"ref": "CurrencyUSD"}
    }
  }
}
```

Use [Look up names and codes](look-things-up.md) to inspect a source or code. The selected source and unit in a response are evidence for the values you receive.

## Next

[How to resolve a question](resolve-a-question.md) shows how to inspect a spec. [How to get a dataset](get-a-dataset.md) shows how to request a version and select a file.
