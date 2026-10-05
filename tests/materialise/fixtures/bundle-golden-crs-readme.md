# OECD Creditor Reporting System (CRS), 1973–2024

Data downloaded from the Development Finance Observatory (DFO), prepared on 4 October 2026.

1,382 rows · 309 variables · 1 donor, 1 recipient · 1973–2024 · Unit: US dollar

## Files

| File | Contents |
| --- | --- |
| `data/oecd-creditor-reporting-system.csv` | The data. One row per observation. |
| `data/codes.csv` | The plain name of every code in the data, such as `country/DEU` for Germany |
| `SOURCES.md` | Who publishes the data, its licence, and each step the DFO applied |
| `CITATION.cff` | The citation, in a format reference managers read |
| `datapackage.json` | A machine-readable description (Frictionless Data Package 2.0) |

## Variables

The file holds 309 variables. The first 10:

- Official development assistance gross disbursements for education facility and training (standard grant)
- Official development assistance gross disbursements for primary education (standard grant)
- Official development assistance gross disbursements for upper secondary schooling (standard grant)
- Official development assistance gross disbursements for vocational training (standard grant)
- Official development assistance gross disbursements for health policy and administrative management (standard grant)
- Official development assistance gross disbursements for medical service (standard grant)
- Official development assistance gross disbursements for basic health care (standard grant)
- Official development assistance gross disbursements for basic health infrastructure (standard grant)
- Official development assistance gross disbursements for basic nutrition (standard grant)
- Official development assistance gross disbursements for mental health and well-being promotion (standard grant)
- and 299 more, all named in `data/codes.csv`.

## Columns

| Column | Meaning |
| --- | --- |
| `stat_var` | The variable measured. Names are in `data/codes.csv`. |
| `financingProvider` | The donor providing the finance. Names are in `data/codes.csv`. |
| `financingRecipient` | The recipient of the finance. Names are in `data/codes.csv`. |
| `time_period` | The period the value refers to. |
| `value` | The observed value, in the unit named in `unit`. |
| `unit` | The unit of `value`. Names are in `data/codes.csv`. |
| `measurementMethod` | How the source measured the value. Empty in this file. |
| `provenance` | The source dataset of the row. Names are in `data/codes.csv`. |
| `status` | The source's flag on the value, such as provisional. Empty in this file. |
| `stat_var_crsCooperationModality` | The CRS co-operation modality of the variable, in words. |
| `stat_var_crsPurpose` | The CRS sector of the variable, in words. |
| `stat_var_crsTypeOfFinance` | The CRS type of finance of the variable, in words. |
| `stat_var_crsTypeOfFlow` | The CRS type of flow of the variable, in words. |

## Licence

Organisation for Economic Co-operation and Development (OECD) publishes OECD Creditor Reporting System (CRS) under OECD Terms and Conditions (https://www.oecd.org/en/about/terms-conditions.html). Those terms govern reuse of this data. The DFO adds no terms of its own.

## How to cite

Cite the source as its publisher asks:

> OECD (2026), Creditor Reporting System (CRS), OECD Data Explorer, https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet (accessed on 2026-10-02)

and this extract:

> Development Finance Observatory (2026). OECD Creditor Reporting System (CRS), 1973–2024 [dataset], version 5c9dfd. https://site.example/data?spec=spec_24e7ab259dcf221c56d7e76edeb977fd&v=cv_5c9dfd37e657d3a3c7000cc6acf49c43

## Get this data again

No API key is needed.

- On the website: https://site.example/data?spec=spec_24e7ab259dcf221c56d7e76edeb977fd&v=cv_5c9dfd37e657d3a3c7000cc6acf49c43. The link keeps working, and this version's rows stay available as Parquet.
- This exact version: `curl "https://site.example/api/datasets/spec_24e7ab259dcf221c56d7e76edeb977fd/cv_5c9dfd37e657d3a3c7000cc6acf49c43"` returns fresh download links. Files fetched only through the API are kept for 30 days from when they were made.
- The latest data for the same definition: `curl "https://site.example/api/datasets/spec_24e7ab259dcf221c56d7e76edeb977fd?format=csv&bundle=true"`.
- To rebuild it yourself, save the request body below as `request.json` and run `curl -X POST -H "Content-Type: application/json" -d @request.json https://site.example/api/datasets`.

```json
{"schema_version": 1, "input": {"kind": "spec_id", "spec_id": "spec_24e7ab259dcf221c56d7e76edeb977fd", "definition": {"schema_version": 1, "spec_id": "spec_24e7ab259dcf221c56d7e76edeb977fd", "identity": {"identity_version": "spec1", "shape": {"kind": "statvar_schema", "population_type": "FinancialFlow", "measured_property": "amount", "stat_type": "measuredValue", "measurement_qualifier": "Nominal"}, "slots": [{"property": "crsCooperationModality", "binding": {"state": "unbound"}}, {"property": "crsPurpose", "binding": {"state": "unbound"}}, {"property": "crsTypeOfFinance", "binding": {"state": "unbound"}}, {"property": "crsTypeOfFlow", "binding": {"state": "value", "ref": "CRSTypeOfFlow10"}}, {"property": "financialFlowType", "binding": {"state": "value", "ref": "Disbursement"}}, {"property": "nettingBasis", "binding": {"state": "value", "ref": "Gross"}}], "entities": {"destination": ["country/NGA"], "origin": ["country/DEU"]}, "requested_window": {"kind": "all"}}}}, "options": {"format": "csv", "bundle": true}}
```

## Open the data

- Spreadsheet: open the CSV in Excel, Numbers or Google Sheets. Look up codes in `data/codes.csv`.
- Python: `pandas.read_csv("data/oecd-creditor-reporting-system.csv")`, or `frictionless.Package("datapackage.json")` for typed columns.
- R: `frictionless::read_package("datapackage.json")`.

## About the Development Finance Observatory

The Development Finance Observatory brings development finance data together in one place. It is an initiative of ONE Data, with support from the Rockefeller Foundation. Find more data at https://site.example. Questions or problems: data@one.org.

## Version

Version 5c9dfd (content version cv_5c9dfd37e657d3a3c7000cc6acf49c43), dataset definition spec_24e7ab259dcf221c56d7e76edeb977fd, prepared 2026-10-04. One version always holds the same numbers.
