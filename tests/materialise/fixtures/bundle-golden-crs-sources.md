# Sources

Who publishes the data in this download, the terms for its reuse, and each step the DFO applied, as the DFO catalogue recorded them when this version was first prepared.

## OECD Creditor Reporting System (CRS)

The Creditor Reporting System (CRS) is the OECD Development Assistance Committee's activity-level database of development finance. Each record is a project, programme or contribution reported by a bilateral provider, a multilateral organisation or a private foundation, with its recipient, sector and purpose, channel of delivery, aid type, and commitment and disbursement amounts. It covers ODA, other official flows, private sector instruments and private philanthropy, and is the source for any breakdown by sector, channel or project.

- Publisher: Organisation for Economic Co-operation and Development (OECD)
- Source page: https://data-explorer.oecd.org/vis?tm=crs&pg=0&snb=30&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_CRS%40DF_CRS&df[ag]=OECD.DCD.FSD&df[vs]=1.6&isAvailabilityDisabled=false&page=0
- Raw data: https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet
- Licence: OECD Terms and Conditions (https://www.oecd.org/en/about/terms-conditions.html)
- Updates: Quarterly
- Last update: 2026-10-02
- Years covered: 1973-2024
- Granularity: Annual, by provider and recipient
- Currency: USD

### How the publisher asks to be cited

> OECD (2026), Creditor Reporting System (CRS), OECD Data Explorer, https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet (accessed on 2026-10-02)

### What the DFO did

1. Download the source data

   The complete CRS file is downloaded from the OECD each time the pipeline runs, and replaces the previous copy, so the data reflects the OECD's latest release.

2. Exclude donor core contributions to multilaterals

   The OECD instructs that donor core contributions to multilateral organisations, reported as ordinary CRS rows flagged `bi_multi = 2`, are excluded to compute bilateral flows. These contributions are already reported in Members' Total Use of the Multilateral System, so keeping them would double count.

3. Standardise providers and recipients

   Provider and recipient codes are standardised to Data Commons IDs used across the platform, so data can be compared across datasets.

4. Select dimensions

   Every figure is reported against the year, the provider and the recipient, and four dimensions from the CRS: type of flow, co-operation modality, type of finance and purpose code. The CRS describes each activity in far more detail than this, including project titles, descriptions, channels and policy markers, none of which is carried through.

5. Aggregate totals

   Activities sharing the same year, provider, recipient and dimensions are summed. Commitments, disbursements, net disbursements and grant equivalents are each aggregated as separate measures. Every figure is reported in US dollars at current prices.

### Version history

- 2026-09-22: Exclude donor core contributions
  - Changed: Donor core contributions to multilateral organisations (`bi_multi = 2`) are excluded from bilateral flows, since they would otherwise double count.
- 2026-08-20: First release
  - Added: Dataset added to the graph.

### Full documentation

- https://site.example/data/sources/provenance/OECDCreditorReportingSystem
- https://site.example/api/graph/documentation/provenance/OECDCreditorReportingSystem
