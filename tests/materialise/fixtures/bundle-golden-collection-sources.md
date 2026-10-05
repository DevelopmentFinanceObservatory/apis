# Sources

Who publishes the data in this download, the terms for its reuse, and each step the DFO applied, as the DFO catalogue recorded them when this version was first prepared.

## IMF World Economic Outlook (WEO)

The World Economic Outlook (WEO) database holds the IMF's macroeconomic data and projections for its member economies and country groups. It covers national accounts, inflation, unemployment, the balance of payments, government revenue, spending and debt, trade and commodity prices, with data from 1980 and projections for most series five years ahead.

- Publisher: International Monetary Fund (IMF)
- Source page: https://data.imf.org/en/datasets/IMF.RES:WEO
- Raw data: https://data.imf.org/en/datasets/IMF.RES:WEO
- Licence: IMF Copyright and Usage (https://www.imf.org/en/about/copyright-and-terms)
- Updates: Semiannual
- Last update: 2026-10-02
- Years covered: 1980-2031

### How the publisher asks to be cited

> Source: International Monetary Fund, World Economic Outlook Database, https://data.imf.org/en/datasets/IMF.RES:WEO

### What the DFO did

1. Download the source data

   The IMF publishes the World Economic Outlook in April and October. Each new release is downloaded from the IMF's data API and replaces the previous copy, so the data reflects the IMF's latest release.

2. Select indicators

   WEO covers national accounts, prices, employment, the balance of payments, government finance and commodity prices. The indicators included cover gross domestic product (GDP), gross national savings, gross capital formation and the output gap.

3. Standardise countries

   Country codes are standardised to Data Commons IDs used across the platform, so data can be compared across datasets. Country groups, such as the world, regions and the euro area, are not included.

4. Convert to whole units

   WEO reports some amounts in billions, which are converted to whole units.

5. Select dimensions

   Every figure is reported against the year and the country. GDP in a country's own currency and in US dollars is published as one variable in two units. Where a country's own currency is the US dollar, the two are the same figure, so only the US-dollar one is published. GDP at purchasing power parity is a separate variable. Shares record what they are a share of, GDP or potential GDP, and per-capita figures are divided by population.

6. Mark projections as estimated

   WEO includes the IMF's projections for the years after the last year the IMF reports for each series. These values are marked as estimated, so they are kept apart from reported values. The purchasing power parity series take the last reported year of GDP in US dollars. The output gap is marked as modelled in every year, since potential output is never observed.

### Version history

- 2026-10-01: First release
  - Added: GDP topic indicators added to the graph.

### Full documentation

- https://site.example/data/sources/provenance/IMFWorldEconomicOutlook
- https://site.example/api/graph/documentation/provenance/IMFWorldEconomicOutlook

## OECD GDP

Gross Domestic Product (GDP) measured by OECD (Organisation for Economic Co-operation and Development).

- Publisher: Organisation for Economic Co-operation and Development (OECD)
- Source page: https://data-explorer.oecd.org/?fs[0]=Topic%2C1%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23&fs[1]=Topic%2C3%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23%7CGDP%20and%20non-financial%20accounts%23ECO_NAD_GNF%23%7CGDP%20and%20components%23ECO_NAD_GNF_GDP%23&pg=0&fc=Topic&snb=21
- Raw data: https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,/Q.....B1GQ......G1.?dimensionAtObservation=AllDimensions
- Licence: Public domain (https://www.oecd.org/termsandconditions/)
- Updates: Annual
- Last update: 2026-10-01

### Full documentation

- https://site.example/data/sources/dc/base/OECDQuarterlyGDP
- https://site.example/api/graph/documentation/dc/base/OECDQuarterlyGDP

## SDG Global Database

- Publisher: United Nations
- Licence: The source declares no licence in the DFO catalogue. Check its terms before reuse.

### Full documentation

- https://site.example/data/sources/undata/p/SDG
- https://site.example/api/graph/documentation/undata/p/SDG

## Unread Source

- Source page: https://unread.example/terms
- Licence: Creative Commons 0 (https://creativecommons.org/publicdomain/zero/1.0/)
