import sqlite3
import pathlib
import multiprocessing
import random
import re
import time
import unicodedata
import openai
import downloaders.scrape_final_reports as scrape_final_reports
import json
import secret_management
from thefuzz import fuzz
from sentence_splitter import split_sentences
import pathlib
import conversions
from antarctic_ladder_metrics.constants import *
import pandas as pd

FINAL_REPORT_PATH = pathlib.Path("data/final_reports")
METRICS_DB_PATH = pathlib.Path("data/final_reports/final_report_metrics_fuzzy.sqlite3")

INTERVENTION_MODEL = "anthropic/claude-sonnet-4.6"

INTERVENTION_INSTRUCTIONS = """You are an assistant for antarctic research. We are trying to figure out which parties are taking an active part in a conversation about antarctic diplomacy. You will be provided with a sentence, and a list of parties. We want you to return a new list, containing all the parties who 'intervened' in the conversation.

A party is said to have intervened in a conversation when they have joined it as an active participant, whether in present or past tense, and not just as a passive object.

Pay attention to the following distinction: a party counts as intervening when the party itself (not merely a named individual affiliated with it) speaks, proposes, objects, reports, or co-sponsors/jointly prepares a document within this conversation. A party does NOT count as intervening when it is mentioned only as the nationality or affiliation of a named individual person, or when it is simply the subject of someone else's praise, thanks, or description for an action taken outside this conversation.

For example, with the following input:
Sentence: China stated that the war in Ukraine halted ice-breaker production, Norway disagreed.
Parties: [China, Ukraine, Norway].

You should respond with: [China, Norway]; since China and Norway both actively participated in the conversation, but Ukraine was only passively mentioned.

Please reply with an array of parties and nothing else."""

# Fill this in with real examples pulled from the corpus. Each entry renders as one more
# worked example appended after INTERVENTION_INSTRUCTIONS, inside the cached prefix shared
# by every classification call - so adding examples here is "free" after the first request
# (cache write), and only costs the 0.1x cache-read rate on every call after that.
#
# {
#     "sentence": "Australia and New Zealand welcomed the report; Chile reserved its position pending review.",
#     "parties": ["Australia", "New Zealand", "Chile"],
#     "intervening": ["Australia", "New Zealand", "Chile"],
# },
FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "sentence": """Bulgaria called on the ATCM to recognise the 
usefulness of the Forum on Education and Outreach and to advise Parties 
to continue to promote Antarctica and Antarctic research in their education 
and public outreach.""",
        "parties": ["Bulgaria"],
        "intervening": ["Bulgaria"],
    },
    {
        "sentence": """The project was intended to directly contribute
to the IPICS Oldest Ice Core Project, which stated the need for multiple ice cores, sharing
the same purpose with France, Italy and Australia.""",
        "parties": ["Australia", "France", "Italy"],
        "intervening": [],
    },
    {
        "sentence": """(8) 
Argentina warmly thanked the individuals involved in preparing the draft 
publication during the intersessional period, including: former CEP Chairs, 
Prof. Olav Orheim of Norway, Dr Tony Press of Australia, Dr Neil Gilbert 
of New Zealand and Dr Yves Frenot of France; current CEP Chair, Mr Ewan 
McIvor; as well as Mr Rodolfo Sánchez of Argentina.""",
        "parties": ["Argentina", "Australia", "France", "New Zealand", "Norway"],
        "intervening": ["Argentina"]
    },
    {
        "sentence": """(403) Spain thanked IAATO for its presentation and its commitment to ensuring transparency
when assessing if its operators were complying with the provisions of the Antarctic Treaty 
and the Environment Protocol.""",
        "parties": ["Spain"],
        "intervening": ["Spain"]
    },
    {
        "sentence": """(20) Australia, in its capacity as Depositary for the Agreement on the Conservation 
of Albatrosses and Petrels (ACAP), reported that there had been no new 
accessions to the Agreement since ATCM XXXVIII, and that there were 
13 Parties to the Agreement (IP 43).""",
        "parties": ["Australia"],
        "intervening": ["Australia"]
    },
    {
        "sentence": """Two weather stations from Korea 
(74°54'01.00"S, 163°43'33.00"E) and China (74°54'04.02"S, 163°43'45.85"E) are present in the Area (see 
Map 2).""",
        "parties": ["Korea", "China"],
        "intervening": []
    },
    {
        "sentence": """(53) The Committee thanked the United Kingdom for its initiative to address the management
of waste of unclear origin, and commended it for its transparency on the issue.""",
        "parties": ["United Kingdom"],
        "intervening": []
    },
    {
        "sentence": """“Within the context of these principles Uruguay proposes, 
through a procedure based on the principle of legal equality, the establishment of a general and 
definitive statute on Antarctica in which, respecting the rights of States as recognized in 
international law, the interests of all States involved and of the international community as a 
whole would be considered equitably.""",
        "parties": ["Uruguay"],
        "intervening": ["Uruguay"]
    },
    {
        "sentence": """ Three IHO Member States (Netherlands, Poland, and Türkiye) have applied to 
become HCA Members in 2023.""",
        "parties": ["Netherlands", "Poland", "Türkiye"],
        "intervening": []
    },
    {
        "sentence": """Ecuador highlighted examples in which spatial 
information on Antarctic geo-objects was being expanded by SCAR and Australia.""",
        "parties": ["Australia", "Ecuador"],
        "intervening": ["Ecuador"]
    },
    {
        "sentence": """(117) The United Kingdom introduced WP 38 Prior assessment of a proposed Antarctic 
Specially Protected Area on Farrier Col, Horseshoe Island, Marguerite Bay, prepared 
jointly with Belgium and Türkiye.""",
        "parties": ["Belgium", "Türkiye", "United Kingdom"],
        "intervening": ["Belgium", "Türkiye", "United Kingdom"],
    },
    {
        "sentence": """(191) Recollecting previous discussions on the matter, China, echoed by Japan, 
expressed the view that matters addressed in the paper needed further 
consideration by the Committee.""",
        "parties": ["China", "Japan"],
        "intervening": ["China", "Japan"],
    }
]


_KEEP_CONTROL_CHARS = {"\n", "\r", "\t"}


def _strip_unicode_artifacts(text: str) -> str:
    """Remove stray PDF-extraction artifacts before text is sent to the LLM: private-use-area
    glyphs (dingbats/bullets baked into a custom PDF font with no real Unicode meaning),
    unassigned codepoints, surrogates, and other control/format characters. Normal whitespace
    (newline/tab/carriage return) and all regular text is preserved untouched."""
    return "".join(
        c for c in text
        if c in _KEEP_CONTROL_CHARS or unicodedata.category(c) not in ("Co", "Cc", "Cf", "Cs", "Cn")
    )


def _render_few_shot_example(example: dict) -> str:
    return (
        f"Sentence: {example['sentence']}\n"
        f"Parties: [{', '.join(example['parties'])}].\n"
        f"Answer: [{', '.join(example['intervening'])}]"
    )


def _render_few_shot_block() -> str:
    if not FEW_SHOT_EXAMPLES:
        return ""
    rendered = "\n\n".join(_render_few_shot_example(example) for example in FEW_SHOT_EXAMPLES)
    return "\n\nMore examples:\n\n" + rendered


# The stable part of the prompt - instructions plus worked examples - shared byte-for-byte
# across every call. This is what gets a cache_control breakpoint.
INTERVENTION_CACHED_PREFIX = _strip_unicode_artifacts(INTERVENTION_INSTRUCTIONS + _render_few_shot_block())


def _format_country_name(country: str) -> str:
    return " ".join(c.capitalize() for c in country.split(" "))


def _openrouter_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=secret_management.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def _build_intervention_messages(sentence: str, parties_display: list[str]) -> list[dict]:
    """Build the chat message with the stable instructions+examples as a cached block,
    and the per-call sentence/parties as a separate, uncached block."""
    sentence = _strip_unicode_artifacts(sentence)
    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": INTERVENTION_CACHED_PREFIX,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Problem:\nSentence: {sentence}\nParties: [{', '.join(parties_display)}].\n",
            },
        ],
    }]


def warm_intervention_cache() -> None:
    """Send one throwaway request so the cached instructions+examples prefix is written
    before the worker pool fans out - otherwise every worker hits a cold cache at once
    and all of them pay the cache-write price instead of just this one call."""
    try:
        _openrouter_client().chat.completions.create(
            model=INTERVENTION_MODEL,
            messages=_build_intervention_messages("Warmup.", ["Chile"]),
            max_tokens=1,
            temperature=1.0,
        )
    except Exception as e:
        print(f"Cache warm-up failed (continuing anyway): {e}")


def classify_intervening_parties(sentence: str, countries: list[str]) -> list[str]:
    """Ask the LLM which of the mentioned countries actively intervened in the sentence.

    Raises on any API failure or unparseable reply, rather than silently treating the
    failure as "nobody intervened" - callers should let the chunk go unprocessed (so it's
    retried on the next run) instead of writing a result for it.
    """
    if not countries:
        return []

    display_names = {country: _format_country_name(country) for country in countries}

    response = _openrouter_client().chat.completions.create(
        model=INTERVENTION_MODEL,
        messages=_build_intervention_messages(sentence, list(display_names.values())),
        temperature=1.0,
    )
    reply = response.choices[0].message.content or ""

    match = re.search(r"\[(.*?)\]", reply, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse intervention list from LLM reply: {reply!r}")

    mentioned = {p.strip().strip('"\'').lower() for p in match.group(1).split(",") if p.strip()}

    return [
        country for country in countries
        if country.lower() in mentioned or display_names[country].lower() in mentioned
    ]

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(METRICS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            document    TEXT    NOT NULL,
            chunk_num   INTEGER NOT NULL,
            chunk       TEXT    NOT NULL,
            year        INTEGER,
            mentioned   TEXT,
            intervening TEXT,
            PRIMARY KEY (document, chunk_num)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def get_document(document: str, chunk_num: int) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT chunk, mentioned, intervening FROM chunks WHERE document=? AND chunk_num=?",
            (document, chunk_num),
        ).fetchone()
    if not row:
        return None
    return {
        "chunk": row[0],
        "mentioned": json.loads(row[1]) if row[1] else [],
        "intervening": json.loads(row[2]) if row[2] else [],
    }


def set_document(
    document: str,
    chunk_num: int,
    chunk: str,
    mentioned: list[str] | None = None,
    intervening: list[str] | None = None,
    year: int | None = None,
) -> None:
    for attempt in range(3):
        try:
            with _get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO chunks (document, chunk_num, chunk, year, mentioned, intervening) VALUES (?, ?, ?, ?, ?, ?)",
                    (document, chunk_num, chunk, year, json.dumps(mentioned or []), json.dumps(intervening or [])),
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
    column = "intervening" if must_be_intervention else "mentioned"
    conditions = [f"{column} LIKE ?"]
    params: list = [f'%"{country}"%']

    if min_year is not None:
        conditions.append("year >= ?")
        params.append(min_year)
    if max_year is not None:
        conditions.append("year <= ?")
        params.append(max_year)

    where = " AND ".join(conditions)
    query = f"SELECT COUNT(*) FROM chunks WHERE {where}"
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
    COUNTRIES = [
            "argentina", "australia", "belgium", "brazil", "bulgaria", "chile", "china",
            "czech republic", "ecuador", "finland", "france", "germany", "india",
            "italy", "japan", "korea", "netherlands", "new zealand",
            "norway", "peru", "poland", "russia", "south africa",
            "spain", "sweden", "turkey", "ukraine", "uruguay", "united kingdom", "united states"
    ]

    def __init__(self):
        with open("data/final_reports/pdf_to_atcm.json", "r") as f:
            pdf_to_atcm_year = json.load(f)

        for pdf in pdf_to_atcm_year:
            pdf_to_atcm_year[pdf] = conversions.actm_meeting_to_year(pdf_to_atcm_year[pdf])

        ocrd_reports = get_ocrd_final_reports()

        to_parse = []
        for base_name in ocrd_reports:
            report_text = ocrd_reports[base_name]
            sentences = split_sentences(report_text)

            for i, chunk in enumerate(sentences):
                if not check_document_exists(base_name, i):
                    to_parse.append((base_name, i, chunk, pdf_to_atcm_year.get(base_name)))

        # Extract mentions up front. Chunks with no country mention at all need no LLM
        # call - write them straight to sqlite3. Only chunks with at least one mention
        # go to the worker pool for intervention classification.
        to_classify = []
        for document, chunk_num, chunk, year in to_parse:
            countries = self.extract_mentions(chunk)
            if not countries:
                set_document(document, chunk_num, chunk, [], [], year)
                continue
            to_classify.append((document, chunk_num, chunk, countries, year))

        if not to_classify:
            return

        # Write the cached prefix once, synchronously, before fanning out - otherwise
        # every one of the 200 workers hits a cold cache simultaneously and all pay the
        # cache-write price instead of just this one call.
        print("Warming Cache")
        warm_intervention_cache()
        print("Cache Warmed")

        with multiprocessing.Pool(200) as pool:
            pool.starmap(self.classify_and_store, to_classify)

    def extract_mentions(self, chunk: str) -> list[str]:
        return fuzzy_term_coincidence_checker(chunk, self.COUNTRIES)

    def classify_and_store(
        self, document: str, chunk_num: int, chunk: str, countries: list[str], year: int | None = None
    ):
        intervening_countries = classify_intervening_parties(chunk, countries)
        set_document(document, chunk_num, chunk, countries, intervening_countries, year)

class FinalReportMentionFigures(FinalReportBaker):
    def __init__(self):
        super().__init__()
        self.yearly_country_to_figure = {}

        for year in range(START_YEAR, END_YEAR+1):
            for country in super().COUNTRIES:
                self.yearly_country_to_figure[(year, " ".join([c.capitalize() for c in country.split(" ")]))] = get_country_figures(country, False, year, year)
        
        self.country_to_figure = {}
        for k in self.yearly_country_to_figure:
            self.country_to_figure[k[1]] = self.country_to_figure.get(k[1], 0) + self.yearly_country_to_figure[k]

    def country_dict(self) -> dict:
        return self.country_to_figure

    def figure_title(self) -> str:
        return "Final Report Mentions"

    def save_full_figures(self, path:str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.yearly_country_to_figure.items()]
        pd.DataFrame(yearly_figures).to_csv(path)\

class FinalReportInterventionFigures(FinalReportBaker):
    def __init__(self):
        super().__init__()
        self.yearly_country_to_figure = {}

        for year in range(START_YEAR, END_YEAR+1):
            for country in super().COUNTRIES:
                self.yearly_country_to_figure[(year, " ".join([c.capitalize() for c in country.split(" ")]))] = get_country_figures(country, True, year, year)
        
        self.country_to_figure = {}
        for k in self.yearly_country_to_figure:
            self.country_to_figure[k[1]] = self.country_to_figure.get(k[1], 0) + self.yearly_country_to_figure[k]

    def country_dict(self) -> dict:
        return self.country_to_figure

    def figure_title(self) -> str:
        return "Final Report Interventions"

    def save_full_figures(self, path:str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.yearly_country_to_figure.items()]
        pd.DataFrame(yearly_figures).to_csv(path)


if __name__ == "__main__":
    FinalReportMentionFigures()