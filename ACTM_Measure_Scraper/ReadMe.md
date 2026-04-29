# ATS Measures Database

## Overview
Adapted from: https://github.com/Parsayarya/ATCM_MeasureDataBase_New

This codebase contains code which:
a) Scrapes ATS measures (and related instruments) from the official webpage
b) Slightly enriches the data with years, based on a list of meeting years for meetings earlier than 1994.

The final processed CSV file is:

- `data/MeasureCorpusEnriched.csv` – the main database ready for analysis and sharing.

---

## How To Run?

To run the full workflow in the current working directory, just run main.py.

To call from another python file, call Pipeline.scrape_and_enrich_measures().

---

## Workflow Summary

### 1. Scraping the Measures

The workflow starts with the scraper:

- `MeasureScraper.py`  

This script loops over the ATS “Measure” pages and, for each document, extracts:

- `Document_Number`
- `Subject`
- `Status` (including adoption text)
- `Category`
- `Topics`
- `Title`
- `Content` (full text where available)

The output of this step is saved as `MeasureCorpus.csv`.

---

### 2. Enrichment

Several variables are extracted from MeasureCorpus.csv.

- `Adoption_Year`
- `ACTM_Number`
- `ACTM_Year`
- `Type`
- `Meeting_Type`

The output of this step is saved as `MeasureCorpusEnriched.csv`

## Final Dataset Columns

The main columns in `MeasureCorpusEnriched.csv` are:

### Identification and Basic Metadata

- **`Document_Number`**  
  ATS document number used on the official website.

- **`Title`**  
  Full official title of the Measure.

- **`Subject`**  
  Short topical label from the ATS page.

- **`Category`**  
  ATS classification (where provided).

- **`Topics`**  
  Additional keywords or thematic labels.

### Status and Timing

- **`Status`**  
  Original status string, typically including information such as whether the Measure is adopted, superseded, or in force, and often the adoption date.

- **`Adoption_Year`**  
  Year extracted from the adoption date embedded in the `Status` text (parsed from `DD/MM/YYYY` where present).

- **`ATCM_Number`**  
  Numeric meeting number derived from Roman numerals in the title.

- **`ATCM_Year`**  
  Year of the relevant ATCM meeting, either parsed directly from the title or filled in from the meeting dictionary.

### Type and Context

- **`Type`**  
  Document type inferred from the start of the title, for example:
  - `Measure`
  - `Decision`
  - `Resolution`
  - `Recommendation`

- **`Meeting_Type`**  
  Broad classification of the meeting context:
  - `ATCM`
  - `SATCM`
  - `CCAS`
  - `Unknown`

### Text Content

- **`Content`**  
  Plain-text body of the Measure (or related instrument), when available.

---