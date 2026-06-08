import scrape_final_reports
import openai
import sqlite3
import pickle
import pathlib
import random
import time
import more_itertools
import asyncio
import secret_management as project_secrets

FINAL_REPORT_PATH = pathlib.Path("data/final_reports")
METRICS_DB_PATH = pathlib.Path("data/final_reports/final_report_metrics.sqlite3")
MENTION = "mention"

_openrouter_client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=project_secrets.get("OPENROUTER_API_KEY"),
)


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(METRICS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentions (
            type      TEXT    NOT NULL,
            document  TEXT    NOT NULL,
            chunk_num INTEGER NOT NULL,
            chunk     TEXT    NOT NULL,
            mentions  TEXT    NOT NULL,
            PRIMARY KEY (type, document, chunk_num)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def db_get(type: str, document: str, chunk_num: int) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT chunk, mentions FROM mentions WHERE type=? AND document=? AND chunk_num=?",
            (type, document, chunk_num),
        ).fetchone()
    return {"chunk": row[0], "mentions": row[1]} if row else None


async def db_set(type: str, document: str, chunk_num: int, chunk: str, mentions: str) -> None:
    for attempt in range(3):
        try:
            with _get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO mentions (type, document, chunk_num, chunk, mentions) VALUES (?, ?, ?, ?, ?)",
                    (type, document, chunk_num, chunk, mentions),
                )
            return
        except sqlite3.OperationalError:
            if attempt == 2:
                raise
            await asyncio.sleep(random.uniform(0.1, 0.5))


def db_check_key_exists(type: str, document: str, chunk_num: int) -> bool:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM mentions WHERE type=? AND document=? AND chunk_num=?",
            (type, document, chunk_num),
        ).fetchone()
    return row is not None

def get_ocrd_final_reports():
    # Activate when OCR DONE.
    # scrape_final_reports.run_final_report_downloading_pipeline()

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


def debug_token_estimate(to_parse: list[tuple[str, str, int]], prompt: str) -> None:
    _tok = lambda s: len(s) // 4
    prompt_tokens = _tok(prompt)
    chunk_tokens = sum(_tok(chunk) for _, chunk, _ in to_parse)
    total = len(to_parse) * prompt_tokens + chunk_tokens
    print(f"chunks:        {len(to_parse)}")
    print(f"prompt tokens: {prompt_tokens} × {len(to_parse)} = {len(to_parse) * prompt_tokens}")
    print(f"chunk tokens (total): {chunk_tokens}")
    print(f"estimated total input tokens: {total}  (~4 chars/tok)")


async def _parse_and_persist_one(base_name: str, chunk: str, chunk_num: int, prompt: str, type: str) -> None:
    response = await _openrouter_client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk},
        ],
    )
    mentions = response.choices[0].message.content
    await db_set(type, base_name, chunk_num, chunk, mentions)


async def parse_and_persist_chunk_info(to_parse: list[tuple[str, str, int]], prompt: str, type: str, concurrency: int = 100) -> None:
    sem = asyncio.Semaphore(concurrency)
    async def _bounded(base_name: str, chunk: str, chunk_num: int) -> None:
        async with sem:
            await _parse_and_persist_one(base_name, chunk, chunk_num, prompt, type)
    await asyncio.gather(*[_bounded(b, c, n) for b, c, n in to_parse])


class FinalReportInterventionFigures:
    INTERVENTION_PROMPT = """You are an assistant for Antarctic research.
    We are attempting to find out how often various countries are said to have "intervened" in ATCM discussions.
    We say that a country intervened in a discussion if it is written that they added something to the conversation.
    For example, "Ukraine stated..." would be an intervention from Ukraine, but not "...when the Ukrainain delegation", "...the Ukranian army...", or "...the Russian invasion of Ukraine..."
    Please return *ONLY* a list of countries which intervened in a discussion segment as a list with no additional commentary. E.g. ["Australia", "Canada"]"""
    INTERVENTION = "intervention"

    def __init__(self):        
        ocrd_reports = get_ocrd_final_reports()

        to_parse = []
        for base_name in ocrd_reports:
            report_text = ocrd_reports[base_name]
            for i, chunk in enumerate(more_itertools.batched(report_text.split('.'), 5)):
                if not db_check_key_exists(self.INTERVENTION, base_name, i):
                    to_parse.append((base_name, '.'.join(chunk), i,))

        print(debug_token_estimate(to_parse, self.INTERVENTION_PROMPT))
        asyncio.run(parse_and_persist_chunk_info(to_parse, self.INTERVENTION_PROMPT, self.INTERVENTION))

class FinalReportMentionFigures:
    def __init__(self):
        pass

if __name__ == "__main__":
    FinalReportInterventionFigures()