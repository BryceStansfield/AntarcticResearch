# ATS Measures Database

## Overview
Adapted from: https://github.com/Parsayarya/ATCM_MeasureDataBase_New

This repository contains the full workflow used to build the Antarctic Treaty System (ATS) Measures database, along with the final cleaned dataset.

The project uses web-scraping, parsing, cleaning, and merging of multiple intermediate files to produce a single, consistent table of Measures (and related instruments) with their adoption details, meeting metadata, and text content.

The final processed CSV file is:

- `MeasureCorpus_withMeetingType3.csv` – the main database ready for analysis and sharing.

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

The output of this step is saved as `MeasureCorpus_Latest_2.csv`. :contentReference[oaicite:0]{index=0}  

---

### 2. Cleaning and Merging Sources

Next, earlier scraped data in `MeasureCorpus_Latest.csv` is combined with the newer file `MeasureCorpus_Latest_2.csv` using:

- `MeasureDBMaker.py` :contentReference[oaicite:1]{index=1}  

In this step:

- Redundant columns such as older `Year` or duplicated `Content` are dropped.
- Both datasets are concatenated into a single corpus.
- An **adoption date** is extracted from the `Status` string (e.g. from `DD/MM/YYYY`) and used to create an `Adoption_Year` column.

This produces `MeasureCorpus_withAdoptionDates_andTypes.csv`.

---

### 3. Building the Meeting Year Dictionary

ATCM meeting years are not always consistently stated in the raw titles, so a separate dictionary of meeting numbers and years is built using:

- `DictionaryForAllMeetings.py` :contentReference[oaicite:2]{index=2}  

This script reads an external ATCM dataset, constructs a `Meeting_Number → Year` mapping, and saves it as `meeting_year_dictionary.csv`.  
This lookup is then used later to fill in missing meeting years.

---

### 4. Final Editing and Classification

The final structural and classification work is done in:

- `MeasureDBEdit.py` :contentReference[oaicite:3]{index=3}  

Key operations here include:

- Extracting **ATCM numbers** from Roman numerals in the title (e.g. `ATCM XXXVIII`).
- Mapping ATCM numbers to years using `meeting_year_dictionary.csv` when the year is missing in the title.
- Identifying the **document type** (`Resolution`, `Decision`, `Measure`, `Recommendation`) from the start of the title.
- Assigning a broader **Meeting_Type** based on text clues in the title (e.g. `ATCM`, `SATCM`, `CCAS`, or `Unknown`).

The result of this step is the final dataset:

- `MeasureCorpus_withMeetingType3.csv`

---

## Final Dataset Columns

The main columns in `MeasureCorpus_withMeetingType3.csv` are:

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

## Reproducing the Workflow

To reproduce the database from scratch:

1. **Scrape the latest Measures**  
   Run `MeasureScraper.py` to generate or update `MeasureCorpus_Latest_2.csv`.

2. **Merge and extract adoption dates**  
   Run `MeasureDBMaker.py` to merge `MeasureCorpus_Latest.csv` and `MeasureCorpus_Latest_2.csv`, and to create `MeasureCorpus_withAdoptionDates_andTypes.csv`.

3. **Build the meeting year dictionary**  
   Run `DictionaryForAllMeetings.py` to produce `meeting_year_dictionary.csv`.

4. **Generate the final database**  
   Run `MeasureDBEdit.py` to add meeting information, document type, and meeting type, and to output `MeasureCorpus_withMeetingType3.csv`.

This final CSV is the recommended file to cite and use for analysis.
