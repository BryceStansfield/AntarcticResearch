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
import pandas as pd
import secret_management
from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations
from sentence_splitter import chunk_sentences

# Working-paper filenames end in a language code (_e, _s, _f, _r); the suite is English.
ENGLISH_SUFFIX = "_e"
DOCUMENT_SUMMARY_PATH = "data/antarctic-db/processed/document-summary.parquet"
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


_authors_cache: dict[str, str] | None = None


def _working_paper_authors() -> dict[str, str]:
    """Map a working paper's filename stem -> its author display string (joined parties),
    from the ATCM WP rows of the document-summary parquet. Memoised."""
    global _authors_cache
    if _authors_cache is None:
        df = pd.read_parquet(DOCUMENT_SUMMARY_PATH)
        df = df[(df["meeting_type"] == "ATCM") & (df["party_type"] == "wp")]
        lookup: dict[str, str] = {}
        for row in df.itertuples():
            if isinstance(row.paper_url, str) and not isinstance(row.parties, float):
                stem = pathlib.Path(row.paper_url).stem
                lookup.setdefault(stem, ", ".join(str(p) for p in row.parties))
        _authors_cache = lookup
    return _authors_cache


def author_for_stem(stem: str) -> str | None:
    """Author display string for a WP filename stem — exact match, then a substring
    fallback for revision-suffix mismatches. When several stems match in the fallback,
    all of their parties are merged into one de-duplicated comma-separated list."""
    authors = _working_paper_authors()
    if stem in authors:
        return authors[stem]
    matches = [a for s, a in authors.items() if s in stem or stem in s]
    if not matches:
        return None
    parties = dict.fromkeys(p.strip() for a in matches for p in a.split(",") if p.strip())
    return ", ".join(parties)


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

LLM_CENSORSHIP_MODEL = "z-ai/glm-5.2"
LLM_CENSORSHIP_DB = pathlib.Path("data/llm_censorship.sqlite3")
# Documents are split into small sentence chunks before going to the LLM, so detection
# (and thus censorship) is localised rather than whole-document.
LLM_CHUNK_SENTENCES = 6
# Where report_top_ratio_chunk dumps the full prompt of the most-censored chunk for testing.
TEST_CHUNK_OUTPUT = pathlib.Path("data/test_censorship_chunk.text")


def _cache_key(author: str, chunk: str) -> str:
    """Cache key for an (author, chunk) pair — the prompt (and thus the answer) depends on
    both, so both are folded into the key."""
    return hashlib.sha256(f"{author}\n{chunk}".encode()).hexdigest()

# Instructions prepended to every passage.
LLM_CENSORSHIP_INSTRUCTIONS = (
    "You are anonymising Working Papers from the Antarctic Treaty Consultative Meeting. "
    "You are given the paper's true author(s) and a passage from it. Find the phrases in the "
    "passage that reveal that THIS author wrote or submitted the paper, so they can be removed.\n\n"
    "A phrase reveals authorship if it ties the paper to whoever wrote it, e.g.:\n"
    "- submission or attribution statements (\"submitted by X\", \"presented by X\", \"prepared by the X delegation\");\n"
    "- first-person references that stand in for the author (\"we\", \"our delegation\", \"our experts\", \"as the host country\").\n"
    "A phrase does NOT reveal authorship — leave it alone — if it merely:\n"
    "- mentions a party in passing or cites another party's work;\n"
    "- refers to a country geographically or discusses it as a topic, even when that country is the author "
    "(a country writing about itself is ordinary subject matter, not an authorship tell).\n\n"
    "Return the shortest identifying span, copied verbatim from the passage — the name, demonym, or "
    "first-person phrase itself (e.g. \"the United Kingdom\", \"British delegation\", \"our delegation\"), not "
    "the surrounding sentence. Reply with ONLY a JSON array of strings; if nothing in the passage reveals "
    "the author, reply with [].\n\n"
    "Example 1:\n"
    "Authors: The United Kingdom\n"
    "Passage: Submitted by the British delegation\n"
    'Answer: ["British"]\n\n'
    "Example 2:\n"
    "Authors: Norway\n"
    "Passage: Icebreaker ships are mainly produced in Norway, the Russian Federation, and the United States\n"
    'Answer: []\n\n'
    "Example 3:\n"
    "Authors: Australia, New Zealand\n"
    "Passage: 1) We the Australian and New Zealand delegations propose an end to penguin trafficking\n"
    'Answer: ["Australian", "New Zealand"]\n\n'
    "Example 4:\n"
    "Authors: Chile\n"
    "Passage: We, the Chilean delegation, hosted the workshop in Punta Arenas, Chile, where Chilean scientists presented their findings.\n"
    'Answer: ["Chilean delegation"]'
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


def _prompt_suffix(author: str, chunk: str) -> str:
    """The per-call (dynamic) tail appended after the shared instruction prefix."""
    return f"\n\nAuthors: {author}\nPassage: {chunk}\n"


def full_prompt(author: str, chunk: str) -> str:
    """The complete prompt text for an (author, chunk) pair — instruction prefix + tail.
    Concatenation of the two cached message parts, for logging / manual model testing."""
    return LLM_CENSORSHIP_INSTRUCTIONS + _prompt_suffix(author, chunk)


def _prompt_content(author: str, chunk: str) -> list[dict]:
    """Message content split at the prompt-cache boundary: the static instruction prefix
    carries a cache_control breakpoint so the provider caches it across calls, while the
    per-chunk tail varies."""
    return [
        {"type": "text", "text": LLM_CENSORSHIP_INSTRUCTIONS, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": _prompt_suffix(author, chunk)},
    ]


def detect_revealing_phrases(author: str, chunk: str) -> list[str]:
    """Ask the LLM for the phrases in a chunk that reveal its (known) authoring party."""
    response = _openrouter_client().chat.completions.create(
        model=LLM_CENSORSHIP_MODEL,
        messages=[{"role": "user", "content": _prompt_content(author, chunk)}],
        # Non-zero so retries (and re-runs) of chunks that returned empty/unparseable have a
        # real chance of a different, parseable reply rather than deterministically repeating.
        temperature=1.0,
        extra_body={"provider": {"order": ["z-ai"], "allow_fallbacks": False}},
    )
    return _parse_phrase_list(response.choices[0].message.content or "")


def warm_llm_cache() -> None:
    """Prime the provider's prompt cache for the shared instruction prefix with a single
    throwaway call, so the parallel detection pool hits a warm cache rather than racing to
    populate it (the prefix dwarfs the per-chunk tail)."""
    try:
        detect_revealing_phrases("Australia", "This paper was submitted by the Australian delegation.")
    except Exception as e:
        print(f"  cache warm failed (continuing): {e}")


def get_or_detect_phrases(author: str, chunk: str) -> list[str]:
    """Cached revealing-phrase lookup for an (author, chunk) pair, calling the LLM on a miss."""
    key = _cache_key(author, chunk)
    cached = get_cached_phrase(key)
    if cached is not None:
        return cached
    phrases = detect_revealing_phrases(author, chunk)
    _store_phrases(key, phrases)
    return phrases


def _detect_with_retry(author: str, chunk: str) -> None:
    """Worker entry point for bulk detection: retries, and on persistent failure leaves the
    chunk uncached (so it's retried next run) rather than aborting the whole pool."""
    for attempt in range(3):
        try:
            get_or_detect_phrases(author, chunk)
            return
        except Exception as e:
            if attempt == 2:
                print(f"  detection failed for {_cache_key(author, chunk)[:12]} (left uncached): {e}")


def detect_all_working_paper_phrases(processes: int = 100) -> None:
    """Send every (uncached) sentence chunk of every English working paper to the LLM and
    cache the revealing-phrase list. Safe to re-run — already-cached chunks are skipped."""
    seen, to_detect = set(), []
    for path in get_working_paper_paths():
        author = author_for_stem(path.stem)
        if author is None:
            continue  # no authorship info — can't fill the prompt
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_sentences(text, LLM_CHUNK_SENTENCES):
            key = _cache_key(author, chunk)
            if key in seen:
                continue
            seen.add(key)
            if get_cached_phrase(key) is None:
                to_detect.append((author, chunk))

    print(f"Detecting revealing phrases for {len(to_detect)} uncached chunks "
          f"(of {len(seen)} total) via {LLM_CENSORSHIP_MODEL}...")
    if not to_detect:
        return
    warm_llm_cache()  # prime the shared-prefix cache before the pool fans out
    with multiprocessing.Pool(processes) as pool:
        pool.starmap(_detect_with_retry, to_detect)


def _cached_phrase_lists() -> dict[str, list[str]]:
    """All cached revealing-phrase lists for the active model, keyed by segment_uuid."""
    with _llm_cache_connection() as conn:
        rows = conn.execute(
            "SELECT segment_uuid, phrases FROM revealing_phrases WHERE model=?",
            (LLM_CENSORSHIP_MODEL,),
        ).fetchall()
    return {uuid: json.loads(phrases) for uuid, phrases in rows}


def flag_heavily_censored_papers(max_censored_chunks: int = 10) -> list[tuple[str, int]]:
    """Flag working papers with more than ``max_censored_chunks`` censored chunks (chunks
    with >=1 revealing phrase) — a sign the LLM over-flagged that paper. For each flagged
    paper, prints the author, the censored/total chunk counts, and each censored chunk's
    phrases. Returns ``(stem, n_censored)`` per flagged paper, most-censored first."""
    cache = _cached_phrase_lists()
    if not cache:
        print("No cached phrase lists yet.")
        return []

    flagged: list[tuple[str, str, list[list[str]]]] = []
    n_papers = 0
    for path in get_working_paper_paths():
        author = author_for_stem(path.stem)
        if author is None:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_sentences(text, LLM_CHUNK_SENTENCES)
        phrase_lists = [cache.get(_cache_key(author, chunk)) for chunk in chunks]
        if all(pl is None for pl in phrase_lists):
            continue  # nothing cached for this paper yet

        n_papers += 1
        censored = [pl for pl in phrase_lists if pl]
        if len(censored) > max_censored_chunks:
            flagged.append((path.stem, author, censored))

    flagged.sort(key=lambda item: len(item[2]), reverse=True)
    print(f"{n_papers} papers — flagging those with > {max_censored_chunks} censored chunks: "
          f"{len(flagged)} papers")
    for stem, author, censored in flagged:
        print(f"  {stem} [author: {author}]: {len(censored)} censored chunks")
        for phrases in censored:
            print(f"    -> {phrases}")
    return [(stem, len(censored)) for stem, _, censored in flagged]


def report_top_ratio_chunk(output_path: pathlib.Path = TEST_CHUNK_OUTPUT) -> tuple[str, str] | None:
    """Find the cached chunk with the largest (#revealing phrases / #authors) ratio — the most
    aggressively censored chunk relative to how many parties authored its paper — and dump that
    chunk's full LLM prompt to ``output_path`` for manual model testing. Returns ``(author,
    chunk)`` of the winner, or ``None`` if nothing is cached."""
    cache = _cached_phrase_lists()
    if not cache:
        print("No cached phrase lists yet.")
        return None

    best = None  # (ratio, n_authors, stem, author, chunk, phrases)
    for path in get_working_paper_paths():
        author = author_for_stem(path.stem)
        if author is None:
            continue
        n_authors = max(1, len([p for p in author.split(",") if p.strip()]))
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_sentences(text, LLM_CHUNK_SENTENCES):
            phrases = cache.get(_cache_key(author, chunk))
            if not phrases:
                continue
            ratio = len(phrases) / n_authors
            if best is None or ratio > best[0]:
                best = (ratio, n_authors, path.stem, author, chunk, phrases)

    if best is None:
        print("No censored chunks cached yet.")
        return None
    ratio, n_authors, stem, author, chunk, phrases = best
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_prompt(author, chunk), encoding="utf-8")
    print(f"Top ratio chunk: {stem} [author: {author}] — "
          f"{len(phrases)} phrases / {n_authors} authors = {ratio:.2f}")
    print(f"  phrases -> {phrases}")
    print(f"  full prompt written to {output_path}")
    return author, chunk


def llm_censor_text(text: str, author: str) -> str:
    """Strip every LLM-identified party-revealing phrase from a document's text. Each sentence
    chunk is judged with the paper's known author in context, and censorship is segment-wide:
    a chunk's phrases are replaced with ``PLACEHOLDER`` only within that chunk (so a phrase
    revealed in one segment never censors another), then the censored chunks are rejoined.
    Run ``detect_all_working_paper_phrases`` first so this is a pure cache read rather than a
    flood of live LLM calls."""
    censored_chunks = []
    for chunk in chunk_sentences(text, LLM_CHUNK_SENTENCES):
        phrases = [p for p in get_or_detect_phrases(author, chunk) if p.strip()]
        if phrases:
            chunk = _censor_pattern(phrases).sub(PLACEHOLDER, chunk)
        censored_chunks.append(chunk)
    return " ".join(censored_chunks) if censored_chunks else text


if __name__ == "__main__":
    # Populate the LLM revealing-phrase cache for every working-paper segment, then report
    # any papers with suspiciously many censored chunks and dump the highest phrase/author
    # ratio chunk's prompt for manual model testing.
    detect_all_working_paper_phrases()
    flag_heavily_censored_papers()
    report_top_ratio_chunk()
