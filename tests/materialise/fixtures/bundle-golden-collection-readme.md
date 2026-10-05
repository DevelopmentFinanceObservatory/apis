# IMF World Economic Outlook (WEO) and 3 more sources, 1980–2031

Data downloaded from the Development Finance Observatory (DFO), prepared on 4 October 2026.

## Files

| File | Contents |
| --- | --- |
| `data/imf-world-economic-outlook-and-oecd-gdp.csv` | IMF World Economic Outlook (WEO) and OECD GDP, 1980–2031 |
| `data/sdg-global-database.csv` | SDG Global Database |
| `data/unread-source.csv` | Unread Source, 2020 |
| `SOURCES.md` | Who publishes the data, its licence, and each step the DFO applied |
| `CITATION.cff` | The citation, in a format reference managers read |
| `datapackage.json` | A machine-readable description (Frictionless Data Package 2.0) |

## Members

- IMF World Economic Outlook (WEO) and OECD GDP, 1980–2031: `data/imf-world-economic-outlook-and-oecd-gdp.csv`, 2,104 rows. Sources: IMF World Economic Outlook (WEO), OECD GDP.
- SDG Global Database: `data/sdg-global-database.csv`. Source: SDG Global Database.
- Unread Source, 2020: `data/unread-source.csv`, 1 row. Source: Unread Source.

## Columns

### `data/imf-world-economic-outlook-and-oecd-gdp.csv`, `data/sdg-global-database.csv`

| Column | Meaning |
| --- | --- |
| `stat_var` | The variable measured. |
| `financingProvider` | The donor providing the finance. |
| `financingRecipient` | The recipient of the finance. |
| `time_period` | The period the value refers to. |
| `value` | The observed value, in the unit named in `unit`. |
| `unit` | The unit of `value`. |
| `measurementMethod` | How the source measured the value. Empty in these files. |
| `provenance` | The source dataset of the row. |
| `status` | The source's flag on the value, such as provisional. Empty in these files. |
| `stat_var_crsCooperationModality` | The CRS co-operation modality of the variable, in words. |
| `stat_var_crsPurpose` | The CRS sector of the variable, in words. |
| `stat_var_crsTypeOfFinance` | The CRS type of finance of the variable, in words. |
| `stat_var_crsTypeOfFlow` | The CRS type of flow of the variable, in words. |

### `data/unread-source.csv`

| Column | Meaning |
| --- | --- |
| `time_period` | The period. |
| `value` | The value. |

## Licence

- International Monetary Fund (IMF) publishes IMF World Economic Outlook (WEO) under IMF Copyright and Usage (https://www.imf.org/en/about/copyright-and-terms).
- Organisation for Economic Co-operation and Development (OECD) publishes OECD GDP in the public domain (https://www.oecd.org/termsandconditions/).
- SDG Global Database: The source declares no licence in the DFO catalogue. Check its terms before reuse.
- Unread Source is published under Creative Commons 0 (https://creativecommons.org/publicdomain/zero/1.0/).

Each source's terms govern reuse of its data. The DFO adds no terms of its own.

## How to cite

Cite each source:

> Source: International Monetary Fund, World Economic Outlook Database, https://data.imf.org/en/datasets/IMF.RES:WEO

> Organisation for Economic Co-operation and Development (OECD), OECD GDP, https://data-explorer.oecd.org/?fs[0]=Topic%2C1%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23&fs[1]=Topic%2C3%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23%7CGDP%20and%20non-financial%20accounts%23ECO_NAD_GNF%23%7CGDP%20and%20components%23ECO_NAD_GNF_GDP%23&pg=0&fc=Topic&snb=21

> United Nations, SDG Global Database

> Unread Source, https://unread.example/terms

and this extract:

> Development Finance Observatory (2026). IMF World Economic Outlook (WEO) and 3 more sources, 1980–2031 [dataset], version 5c9dfd. https://site.example/api/datasets/col_7d41b2000000000000000000000000/cv_5c9dfd37e657d3a3c7000cc6acf49c43

## Get this data again

No API key is needed.

- This exact version: `curl "https://site.example/api/datasets/col_7d41b2000000000000000000000000/cv_5c9dfd37e657d3a3c7000cc6acf49c43"` returns fresh download links. Files fetched only through the API are kept for 30 days from when they were made.
- The latest data for the same definition: `curl "https://site.example/api/datasets/col_7d41b2000000000000000000000000?format=csv&bundle=true"`.

## Open the data

- Spreadsheet: open the CSV in Excel, Numbers or Google Sheets.
- Python: `pandas.read_csv("data/imf-world-economic-outlook-and-oecd-gdp.csv")`, or `frictionless.Package("datapackage.json")` for typed columns.
- R: `frictionless::read_package("datapackage.json")`.

## About the Development Finance Observatory

The Development Finance Observatory brings development finance data together in one place. It is an initiative of ONE Data, with support from the Rockefeller Foundation. Find more data at https://site.example. Questions or problems: data@one.org.

## Version

Version 5c9dfd (content version cv_5c9dfd37e657d3a3c7000cc6acf49c43), collection definition col_7d41b2000000000000000000000000, prepared 2026-10-04. One version always holds the same numbers.
