"""Filter fluff sentences out of (already censored) working papers.

Where ``working_paper_censorship`` strips party-revealing phrases, this module drops whole
sentences that carry no diplomatic substance. A censored document is broken into single
sentences, each sentence is hashed, and an LLM classifies it as semantically IMPORTANT (it
conveys a stance about Antarctic diplomacy) or FLUFF (boilerplate / procedural filler that
takes no position). The per-sentence labels are cached in sqlite so the (paid) classification
only runs once, keyed on the sentence hash and the model.

Run ``detect_all_working_paper_phrases`` (from ``working_paper_censorship``) first so the
censorship step is a pure cache read, then ``classify_all_working_paper_sentences`` here.
"""
import sys
import time
import random
import pathlib
import sqlite3
import hashlib
import multiprocessing

import openai
import secret_management
from sentence_splitter import split_sentences
from embeddings.working_paper_censorship import (
    get_working_paper_paths,
    author_for_stem,
    llm_censor_text,
)

SEMANTIC_FILTER_MODEL = "openai/gpt-oss-120b"
SEMANTIC_FILTER_DB = pathlib.Path("data/semantic_filter.sqlite3")
# Where annotate_working_paper writes its tagged debug copies.
DEBUG_OUTPUT_DIR = pathlib.Path("data/semantic_filter_debug")

# Instructions prepended to every sentence.
SEMANTIC_FILTER_INSTRUCTIONS = (
    "You are filtering the sentences of Working Papers from the Antarctic Treaty Consultative "
    "Meeting, keeping only those that carry diplomatic substance.\n\n"
    "Classify a SINGLE sentence as one of:\n"
    "- IMPORTANT: it conveys a STANCE on an issue of Antarctic diplomacy — a position, proposal, "
    "recommendation, concern, commitment, objection, or substantive argument about how Antarctica "
    "should be governed, protected, used, or regulated.\n"
    "- FLUFF: it is boilerplate or procedural filler that takes no position — greetings, section "
    "headings, agenda-item or attachment references, acknowledgements, restating a meeting's title, "
    "table-of-contents lines, or purely descriptive background that argues for nothing.\n\n"
    "Reply with ONLY one word: IMPORTANT or FLUFF.\n\n"
    "Example 1:\n"
    "Sentence: This paper is submitted under Agenda Item 7.\n"
    "Answer: FLUFF\n\n"
    "Example 2:\n"
    "Sentence: We propose that all vessels operating south of 60S be required to carry an ice pilot.\n"
    "Answer: IMPORTANT\n\n"
    "Example 3:\n"
    "Sentence: The Antarctic Treaty was signed in 1959.\n"
    "Answer: FLUFF\n\n"
    "Example 4:\n"
    "Sentence: CountryName strongly opposes any expansion of krill fishing in the Ross Sea region.\n"
    "Answer: IMPORTANT"
)


def _cache_key(sentence: str) -> str:
    return hashlib.sha256(sentence.encode()).hexdigest()


def _openrouter_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=secret_management.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def _cache_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(SEMANTIC_FILTER_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentence_importance (
            sentence_hash TEXT NOT NULL,
            model         TEXT NOT NULL,
            important     INTEGER NOT NULL,
            PRIMARY KEY (sentence_hash, model)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def get_cached_label(sentence_hash: str, model: str = SEMANTIC_FILTER_MODEL) -> bool | None:
    with _cache_connection() as conn:
        row = conn.execute(
            "SELECT important FROM sentence_importance WHERE sentence_hash=? AND model=?",
            (sentence_hash, model),
        ).fetchone()
    return bool(row[0]) if row else None


def _store_label(sentence_hash: str, important: bool, model: str = SEMANTIC_FILTER_MODEL) -> None:
    for attempt in range(3):
        try:
            with _cache_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sentence_importance "
                    "(sentence_hash, model, important) VALUES (?, ?, ?)",
                    (sentence_hash, model, int(important)),
                )
            return
        except sqlite3.OperationalError:
            if attempt == 2:
                raise
            time.sleep(random.uniform(0.1, 0.5))


def _parse_label(reply: str) -> bool:
    """True if the reply says IMPORTANT, False if FLUFF. Raises if the reply names neither (or
    both) so the caller can retry rather than cache a guess."""
    text = reply.upper()
    important, fluff = "IMPORTANT" in text, "FLUFF" in text
    if important == fluff:
        raise ValueError(f"Could not parse IMPORTANT/FLUFF from LLM reply: {reply!r}")
    return important


def _prompt(sentence: str) -> str:
    """The full prompt for a sentence — shared instructions plus the per-sentence tail."""
    return f"{SEMANTIC_FILTER_INSTRUCTIONS}\n\nSentence: {sentence}\nAnswer:"


def classify_sentence(sentence: str) -> bool:
    """Ask the LLM whether a single sentence is semantically important. Returns True (IMPORTANT)
    or False (FLUFF)."""
    response = _openrouter_client().chat.completions.create(
        model=SEMANTIC_FILTER_MODEL,
        messages=[{"role": "user", "content": _prompt(sentence)}],
        # Non-zero so retries of unparseable replies have a real chance of a different, parseable
        # answer rather than deterministically repeating (hosted providers aren't deterministic).
        temperature=1.0,
    )
    return _parse_label(response.choices[0].message.content or "")


def get_or_classify(sentence: str) -> bool:
    """Cached importance lookup for a sentence, calling the LLM on a miss."""
    key = _cache_key(sentence)
    cached = get_cached_label(key)
    if cached is not None:
        return cached
    important = classify_sentence(sentence)
    _store_label(key, important)
    return important


def _classify_with_retry(sentence: str) -> None:
    """Worker entry point for bulk classification: retries, and on persistent failure leaves the
    sentence uncached (so it's retried next run) rather than aborting the whole pool."""
    for attempt in range(3):
        try:
            get_or_classify(sentence)
            return
        except Exception as e:
            if attempt == 2:
                print(f"  classification failed for {_cache_key(sentence)[:12]} (left uncached): {e}")


def _iter_censored_sentences():
    """Yield every non-empty sentence of every English working paper, after censorship. Relies on
    the censorship phrase cache being populated (run ``detect_all_working_paper_phrases`` first),
    otherwise ``llm_censor_text`` fires live censorship calls here."""
    for path in get_working_paper_paths():
        author = author_for_stem(path.stem)
        if author is None:
            continue  # no authorship info — can't censor, so skip
        text = path.read_text(encoding="utf-8", errors="ignore")
        for sentence in split_sentences(llm_censor_text(text, author)):
            sentence = sentence.strip()
            if sentence:
                yield sentence


def classify_all_working_paper_sentences(processes: int = 100) -> None:
    """Classify every (uncached) censored sentence of every English working paper and cache the
    IMPORTANT/FLUFF label. Safe to re-run — already-cached sentences are skipped."""
    seen, to_classify = set(), []
    for sentence in _iter_censored_sentences():
        key = _cache_key(sentence)
        if key in seen:
            continue
        seen.add(key)
        if get_cached_label(key) is None:
            to_classify.append(sentence)

    print(f"Classifying {len(to_classify)} uncached sentences (of {len(seen)} unique) "
          f"via {SEMANTIC_FILTER_MODEL}...")
    if not to_classify:
        return
    with multiprocessing.Pool(processes) as pool:
        pool.map(_classify_with_retry, to_classify)


def filter_text(text: str, author: str) -> str:
    """Return ``text`` with fluff sentences dropped: censor it, split into sentences, and keep only
    the ones classified IMPORTANT (rejoined with spaces). Run the bulk passes first so this is a
    pure cache read rather than a flood of live LLM calls."""
    kept = [
        sentence
        for raw in split_sentences(llm_censor_text(text, author))
        if (sentence := raw.strip()) and get_or_classify(sentence)
    ]
    return " ".join(kept)


def report_filter_stats() -> None:
    """Print how many cached sentences are IMPORTANT vs FLUFF for the active model."""
    with _cache_connection() as conn:
        rows = conn.execute(
            "SELECT important, COUNT(*) FROM sentence_importance WHERE model=? GROUP BY important",
            (SEMANTIC_FILTER_MODEL,),
        ).fetchall()
    counts = {bool(important): n for important, n in rows}
    n_important, n_fluff = counts.get(True, 0), counts.get(False, 0)
    total = n_important + n_fluff
    if not total:
        print("No cached sentence labels yet.")
        return
    print(f"{total} cached sentences [{SEMANTIC_FILTER_MODEL}]: "
          f"{n_important} important ({n_important / total:.1%}), "
          f"{n_fluff} fluff ({n_fluff / total:.1%})")


def _resolve_working_paper(stem_or_path: str | pathlib.Path) -> pathlib.Path:
    """Resolve a filename stem (e.g. ``ATCM16_wp001_e``) or a path to a working-paper text file."""
    path = pathlib.Path(stem_or_path)
    if path.exists():
        return path
    stem = path.stem
    for wp in get_working_paper_paths():
        if wp.stem == stem:
            return wp
    raise FileNotFoundError(f"No working paper found for {stem_or_path!r}")


def annotate_working_paper(
    stem_or_path: str | pathlib.Path, output_path: pathlib.Path | None = None
) -> pathlib.Path:
    """Debug helper: censor a working paper, classify each sentence, and write a copy with every
    sentence tagged ``[KEEP ]`` (important) or ``[FLUFF]`` on its own line. Reads the
    sentence-importance cache, classifying live on a miss. ``stem_or_path`` accepts a filename stem
    or a path; ``output_path`` defaults to ``DEBUG_OUTPUT_DIR/<stem>_annotated.txt``. Returns the
    path written."""
    path = _resolve_working_paper(stem_or_path)
    author = author_for_stem(path.stem)
    if author is None:
        print(f"  warning: no authorship info for {path.stem} — annotating without censorship")
    text = path.read_text(encoding="utf-8", errors="ignore")
    censored = llm_censor_text(text, author) if author is not None else text

    tagged, n_fluff = [], 0
    for raw in split_sentences(censored):
        sentence = raw.strip()
        if not sentence:
            continue
        important = get_or_classify(sentence)
        n_fluff += not important
        tagged.append(f"[{'KEEP ' if important else 'FLUFF'}] {sentence}")

    n_kept = len(tagged) - n_fluff
    if output_path is None:
        output_path = DEBUG_OUTPUT_DIR / f"{path.stem}_annotated.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# {path.stem}  author={author!r}  model={SEMANTIC_FILTER_MODEL}\n"
        f"# {len(tagged)} sentences: {n_kept} kept / {n_fluff} fluff\n\n"
    )
    output_path.write_text(header + "\n".join(tagged) + "\n", encoding="utf-8")
    print(f"Annotated {path.stem}: {n_kept} kept / {n_fluff} fluff -> {output_path}")
    return output_path


if __name__ == "__main__":
    # With a working-paper stem/path argument, write a fluff-annotated debug copy of that paper.
    # Otherwise populate the sentence-importance cache for every censored working-paper sentence,
    # then report the important/fluff split.
    if len(sys.argv) > 1:
        annotate_working_paper(sys.argv[1])
    else:
        classify_all_working_paper_sentences()
        report_filter_stats()
