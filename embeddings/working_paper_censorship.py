"""Censor party-identifying information from working papers.

Two strategies are offered:
- ``censor_text`` (naive): replace every mention of a target country's name with a neutral
  placeholder ("CountryName").
- ``llm_censor_text`` (llm_censorship): ask an LLM, per small sentence chunk, for the exact
  phrases that reveal who authored the paper, then strip those phrases. The per-chunk
  phrase lists are cached in sqlite so the (paid) detection only runs once.
"""
import re
import json
import time
import random
import pathlib
import sqlite3
import hashlib
import multiprocessing

import openai
import secret_management
from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations
from sentence_splitter import chunk_sentences

# Working-paper filenames end in a language code (_e, _s, _f, _r); the suite is English.
ENGLISH_SUFFIX = "_e"
# Default countries to censor — the authorship targets.
COUNTRIES = ["Australia", "Australian", "the United Kingdom", "United Kingdom", "the UK", "British", "the USA", "the US", "United States", "American", "Norway", "Norwegian", "Chile", "Chilean"]
PLACEHOLDER = "CountryName"


def get_working_paper_paths() -> list[pathlib.Path]:
    """All unique English working-paper text files (those under a /wp/ directory)."""
    locations = map_all_wp_ip_file_locations()
    paths = {
        pathlib.Path(p)
        for p in locations.values()
        if "/wp/" in p and pathlib.Path(p).stem.endswith(ENGLISH_SUFFIX)
    }
    return sorted(paths)


def _censor_pattern(countries: list[str]) -> re.Pattern:
    """Whole-phrase, case-insensitive matcher for the supplied country names, tolerant of
    OCR whitespace / line breaks. Longest names first so multi-word names take precedence."""
    bodies = [
        r"\s+".join(re.escape(word) for word in country.split())
        for country in sorted(countries, key=len, reverse=True)
    ]
    return re.compile(r"\b(?:" + "|".join(bodies) + r")\b", re.IGNORECASE)


def censor_text(text: str, countries: list[str] = COUNTRIES) -> str:
    """Replace every mention of a supplied country name with ``PLACEHOLDER``."""
    if not countries:
        return text
    return _censor_pattern(countries).sub(PLACEHOLDER, text)


# --------------------------------------------------------------------------- LLM censorship

LLM_CENSORSHIP_MODEL = "deepseek/deepseek-v4-flash"
LLM_CENSORSHIP_DB = pathlib.Path("data/llm_censorship.sqlite3")
# Documents are split into small sentence chunks before going to the LLM, so detection
# (and thus censorship) is localised rather than whole-document.
LLM_CHUNK_SENTENCES = 6


def _llm_chunks(text: str) -> list[tuple[str, str]]:
    """Split a document into fixed-size sentence chunks as (chunk_hash, chunk_text) pairs."""
    return [(hashlib.sha256(chunk.encode()).hexdigest(), chunk)
            for chunk in chunk_sentences(text, LLM_CHUNK_SENTENCES)]

# Instructions prepended to every passage.
LLM_CENSORSHIP_INSTRUCTIONS = (
    "You are anonymising Working Papers from the Antarctic Treaty Consultative Meeting. "
    "Given a passage from a Working Paper, identify the exact phrases in it that reveal "
    "which party (country, delegation, or organisation) authored or submitted the paper. "
    "Only list phrases that directly reveal who wrote the working paper, not ones that "
    "merely reference geography or mention a party. "
    "Copy each phrase verbatim from the passage. Reply with ONLY a JSON array of strings; "
    "if nothing in the passage reveals the author, reply with [].\n\n"
    "Example 1:\n"
    "Passage: Submitted by the United Kingdom delegation\n"
    'Answer: ["United Kingdom"]\n\n'
    "Example 2: \n"
    "Passage: The culling of penguins in McMurdo Station by Chile and Norway must cease\n"
    'Answer: []'
)


def _openrouter_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=secret_management.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def _llm_cache_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(LLM_CENSORSHIP_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS revealing_phrases (
            segment_uuid TEXT NOT NULL,
            model        TEXT NOT NULL,
            phrases      TEXT NOT NULL,
            PRIMARY KEY (segment_uuid, model)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def get_cached_phrase(segment_uuid: str, model: str = LLM_CENSORSHIP_MODEL) -> list[str] | None:
    with _llm_cache_connection() as conn:
        row = conn.execute(
            "SELECT phrases FROM revealing_phrases WHERE segment_uuid=? AND model=?",
            (segment_uuid, model),
        ).fetchone()
    return json.loads(row[0]) if row else None


def _store_phrases(segment_uuid: str, phrases: list[str], model: str = LLM_CENSORSHIP_MODEL) -> None:
    for attempt in range(3):
        try:
            with _llm_cache_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO revealing_phrases (segment_uuid, model, phrases) VALUES (?, ?, ?)",
                    (segment_uuid, model, json.dumps(phrases)),
                )
            return
        except sqlite3.OperationalError:
            if attempt == 2:
                raise
            time.sleep(random.uniform(0.1, 0.5))


def _parse_phrase_list(reply: str) -> list[str]:
    match = re.search(r"\[.*\]", reply, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse phrase list from LLM reply: {reply!r}")
    return [str(p).strip() for p in json.loads(match.group(0)) if str(p).strip()]


def detect_revealing_phrases(segment_text: str) -> list[str]:
    """Ask the LLM for the phrases in a segment that reveal its authoring party."""
    response = _openrouter_client().chat.completions.create(
        model=LLM_CENSORSHIP_MODEL,
        messages=[{"role": "user", "content": f"{LLM_CENSORSHIP_INSTRUCTIONS}\n\nPassage:\n{segment_text}\n"}],
        temperature=0.0,
    )
    return _parse_phrase_list(response.choices[0].message.content or "")


def get_or_detect_phrases(segment_uuid: str, segment_text: str) -> list[str]:
    """Cached revealing-phrase lookup for a segment, calling the LLM only on a miss."""
    cached = get_cached_phrase(segment_uuid)
    if cached is not None:
        return cached
    phrases = detect_revealing_phrases(segment_text)
    _store_phrases(segment_uuid, phrases)
    return phrases


def _detect_with_retry(segment_uuid: str, segment_text: str) -> None:
    """Worker entry point for bulk detection: retries, and on persistent failure leaves the
    segment uncached (so it's retried next run) rather than aborting the whole pool."""
    for attempt in range(3):
        try:
            get_or_detect_phrases(segment_uuid, segment_text)
            return
        except Exception as e:
            if attempt == 2:
                print(f"  detection failed for {segment_uuid[:12]} (left uncached): {e}")


def detect_all_working_paper_phrases(processes: int = 100) -> None:
    """Send every (uncached) sentence chunk of every English working paper to the LLM and
    cache the revealing-phrase list. Safe to re-run — already-cached chunks are skipped."""
    seen, to_detect = set(), []
    for path in get_working_paper_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk_uuid, chunk in _llm_chunks(text):
            if chunk_uuid in seen:
                continue
            seen.add(chunk_uuid)
            if get_cached_phrase(chunk_uuid) is None:
                to_detect.append((chunk_uuid, chunk))

    print(f"Detecting revealing phrases for {len(to_detect)} uncached chunks "
          f"(of {len(seen)} total) via {LLM_CENSORSHIP_MODEL}...")
    if not to_detect:
        return
    with multiprocessing.Pool(processes) as pool:
        pool.starmap(_detect_with_retry, to_detect)


def flag_long_phrase_lists(max_phrases: int = 10) -> list[tuple[str, list[str]]]:
    """Flag segments whose revealing-phrase list has more than ``max_phrases`` entries —
    usually a sign the LLM over-flagged."""
    with _llm_cache_connection() as conn:
        rows = conn.execute(
            "SELECT segment_uuid, phrases FROM revealing_phrases WHERE model=?",
            (LLM_CENSORSHIP_MODEL,),
        ).fetchall()
    if not rows:
        print("No cached phrase lists yet.")
        return []

    lists = [(uuid, json.loads(phrases)) for uuid, phrases in rows]
    flagged = sorted((item for item in lists if len(item[1]) > max_phrases),
                     key=lambda item: len(item[1]), reverse=True)
    print(f"{len(lists)} segments — flagging lists with > {max_phrases} phrases: {len(flagged)} segments")
    for uuid, phrases in flagged:
        print(f"  {uuid[:12]}: {len(phrases)} phrases -> {phrases}")
    return flagged


def llm_censor_text(text: str) -> str:
    """Strip every LLM-identified party-revealing phrase from a document's text. Phrases are
    gathered (from cache, detecting on miss) across the document's sentence chunks, then
    replaced with ``PLACEHOLDER`` throughout. Run ``detect_all_working_paper_phrases`` first
    so this is a pure cache read rather than a flood of live LLM calls."""
    phrases = set()
    for chunk_uuid, chunk in _llm_chunks(text):
        phrases.update(get_or_detect_phrases(chunk_uuid, chunk))
    phrases = sorted(p for p in phrases if p.strip())
    if not phrases:
        return text
    return _censor_pattern(phrases).sub(PLACEHOLDER, text)


if __name__ == "__main__":
    # Populate the LLM revealing-phrase cache for every working-paper segment, then report
    # any segments with suspiciously long phrase lists.
    detect_all_working_paper_phrases()
    flag_long_phrase_lists()
