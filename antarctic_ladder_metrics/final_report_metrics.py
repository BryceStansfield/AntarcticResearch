import sqlite3
import pathlib
import random
import more_itertools
import asyncio
import time
import nltk
import downloaders.scrape_final_reports as scrape_final_reports
import json
from thefuzz import fuzz
from nltk.tokenize import sent_tokenize
import pathlib
import conversions
import country_meta_info

# TODO: Use LLMs for intervention filtering.

FINAL_REPORT_PATH = pathlib.Path("data/final_reports")
METRICS_DB_PATH = pathlib.Path("data/final_reports/final_report_metrics_fuzzy.sqlite3")

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(METRICS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            document  TEXT    NOT NULL,
            chunk_num INTEGER NOT NULL,
            chunk     TEXT    NOT NULL,
            year      INTEGER,
            PRIMARY KEY (document, chunk_num)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentions_and_interventions (
            document        TEXT    NOT NULL,
            chunk_num       INTEGER NOT NULL,
            country         TEXT    NOT NULL,
            is_intervention INTEGER NOT NULL,
            PRIMARY KEY (document, chunk_num, country)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def get_document(document: str, chunk_num: int) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT chunk FROM chunks WHERE document=? AND chunk_num=?",
            (document, chunk_num),
        ).fetchone()
    return {"chunk": row[0]} if row else None


def set_document(document: str, chunk_num: int, chunk: str, year: int | None = None) -> None:
    for attempt in range(3):
        try:
            with _get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO chunks (document, chunk_num, chunk, year) VALUES (?, ?, ?, ?)",
                    (document, chunk_num, chunk, year),
                )
            return
        except sqlite3.OperationalError:
            if attempt == 2:
                raise
            time.sleep(random.uniform(0.1, 0.5))


def set_mentions_and_interventions(
    document: str, chunk_num: int, countries: list[str], is_intervention: bool
) -> None:
    for attempt in range(3):
        try:
            with _get_connection() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO mentions_and_interventions (document, chunk_num, country, is_intervention) VALUES (?, ?, ?, ?)",
                    [(document, chunk_num, country, int(is_intervention)) for country in countries],
                )
            return
        except sqlite3.OperationalError:
            if attempt == 2:
                raise
            time.sleep(random.uniform(0.1, 0.5))

def check_document_exists(document: str, chunk_num: int) -> bool:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM chunks WHERE document=? AND chunk_num=?",
            (document, chunk_num),
        ).fetchone()
    return row is not None

def get_country_figures(country: str, must_be_intervention: bool, min_year: int | None = None, max_year: int | None = None):
    conditions = ["m.country = ?"]
    params: list = [country]

    if must_be_intervention:
        conditions.append("m.is_intervention")
    if min_year is not None:
        conditions.append("c.year >= ?")
        params.append(min_year)
    if max_year is not None:
        conditions.append("c.year <= ?")
        params.append(max_year)

    where = " AND ".join(conditions)
    query = f"""
        SELECT COUNT(*) FROM mentions_and_interventions m
        JOIN chunks c ON m.document = c.document AND m.chunk_num = c.chunk_num
        WHERE {where}
    """
    with _get_connection() as conn:
        return conn.execute(query, params).fetchone()[0]

def get_ocrd_final_reports():
    scrape_final_reports.run_final_report_downloading_pipeline()

    paths = []
    for path in FINAL_REPORT_PATH.iterdir():
        if not path.is_file() or path.suffix != ".txt":
            continue
        paths.append(path)

    ocr_report_dict = {}
    for path in paths:
        with open(path, "r") as f:
            ocr_report_dict[path.name.split('.ocr')[0]] = f.read()
    
    return ocr_report_dict

def fuzzy_term_coincidence_checker(sentence: str, terms: list[str]):
    sentence = sentence.lower()

    return list(filter(lambda c: fuzz.partial_ratio(c, sentence) >= 90, terms))


class FinalReportBaker:
    INTERVENTION_TERMS = ['stated', 'noted', "declared", "expressed", "mentioned", "announced", "asserted", "conveyed", "disclosed", "observed", "recorded", "highlighted", "pointed out", "acknowledged", "recognized", "reported"]
    COUNTRIES = [
            "argentina", "australia", "belgium", "brazil", "bulgaria", "chile", "china",
            "czech republic", "ecuador", "finland", "france", "germany", "india",
            "Italy", "japan", "korea", "netherlands", "new zealand",
            "norway", "peru", "poland", "russia", "south africa",
            "spain", "sweden", "ukraine", "uruguay", "united kingdom", "united states"
    ]

    def __init__(self):
        with open("data/final_reports/pdf_to_atcm.json", "r") as f:
            pdf_to_atcm_year = json.load(f)

        for pdf in pdf_to_atcm_year:
            pdf_to_atcm_year[pdf] = conversions.actm_meeting_to_year(pdf_to_atcm_year[pdf])

        nltk.download('punkt_tab')
        ocrd_reports = get_ocrd_final_reports()

        to_parse = []
        for base_name in ocrd_reports:
            report_text = ocrd_reports[base_name]
            sentences = sent_tokenize(report_text)

            for i, chunk in enumerate(sentences):
                if not check_document_exists(base_name, i):
                    to_parse.append((base_name, chunk, i,))

        for t in to_parse:
            self.add_mentions_and_interventions(t[0], t[2], t[1], pdf_to_atcm_year.get(t[0]))

    def add_mentions_and_interventions(self, document: str, chunk_num: int, chunk: str, year: int | None = None):
        countries = fuzzy_term_coincidence_checker(chunk, self.COUNTRIES)
        is_intervention = len(fuzzy_term_coincidence_checker(chunk, self.INTERVENTION_TERMS)) > 0

        set_mentions_and_interventions(document, chunk_num, countries, is_intervention)
        set_document(document, chunk_num, chunk, year)

class FinalReportMentionFigures(FinalReportBaker):
    def __init__(self):
        super().__init__()
        self.country_to_figure = {}

        for country in super().COUNTRIES:
            self.country_to_figure[" ".join([c.capitalize() for c in country.split(" ")])] = get_country_figures(country, False, 2000, 2024)

    def country_dict(self) -> dict:
        return self.country_to_figure

    def figure_title(self) -> str:
        return "Final Report Mentions"


class FinalReportInterventionFigures(FinalReportBaker):
    def __init__(self):
        super().__init__()
        self.country_to_figure = {}

        for country in super().COUNTRIES:
            self.country_to_figure[" ".join([c.capitalize() for c in country.split(" ")])] = get_country_figures(country, True, 2000, 2024)

    def country_dict(self) -> dict:
        return self.country_to_figure

    def figure_title(self) -> str:
        return "Final Report Interventions"


if __name__ == "__main__":
    FinalReportMentionFigures()
    FinalReportMentionFigures()