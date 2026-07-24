"""A cached cross-encoder reranker for working papers, via OpenRouter.

``rerank_latency_comparison.py`` wants a second opinion on "which working paper
best matches this instrument" that does *not* come from the same Qwen embedding
space the nearest-neighbour search already used. A cross-encoder reranker reads
the query and each document together rather than comparing two independently
produced vectors, so it can disagree with cosine order in ways worth looking at.

Model
-----
``cohere/rerank-4-pro`` through OpenRouter's ``/api/v1/rerank`` endpoint (the
same key that powers the embeddings, ``OPENROUTER_API_KEY``). Its request/
response follow the Cohere rerank shape: ``{model, query, documents, top_n}`` in,
``{"results": [{"index", "relevance_score"}, ...]}`` out, where ``index`` points
back into the ``documents`` array that was sent.

Why per-pair caching works
--------------------------
A cross-encoder scores each (query, document) pair on its own -- the relevance of
one document does not depend on which other documents share the batch. So a
score is a pure function of (model, query, document) and can be cached at that
granularity, independent of the set it was requested in. That makes the cache
reusable across overlapping candidate sets and lets a repeat call touch the
network only for pairs it has never seen. The cache key is a sha256 over
``model \x1f query \x1f document`` using the *exact* strings sent (documents are
truncated to ``MAX_DOC_CHARS`` first), so a hit always reflects the same request.

Documents are truncated to ``MAX_DOC_CHARS`` characters, sized to fill the
model's 32K-token window rather than a fraction of it, since a full OCR'd working
paper can still overrun that window; see the constant for the sizing.
"""

import hashlib
import sqlite3
import time

import requests

import secret_management

RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_MODEL = "cohere/rerank-4-pro"

# Fill the model's 32K-token window rather than a token or two of it: the cosine
# neighbour this rerank is judged against was computed over the *whole* document
# (mean-pooled across every segment), so the cross-encoder should see as much of
# the paper as it can hold. At the pipeline's ~3 chars/token heuristic (the same
# one split_long_document uses) 32K tokens is ~96K chars; 90K leaves headroom for
# the short instrument query, which shares the window. The rare paper longer than
# this loses only its tail -- the subject and body a rerank turns on sit up front.
MAX_DOC_CHARS = 90000

_CACHE_PATH = "data/latencies/rerank_cache.sqlite3"
_SEP = "\x1f"  # unit separator; never appears in the texts, so keys can't collide


def _get_cache() -> sqlite3.Connection:
    # A fresh connection per call, so callers may fan out over threads. The
    # timeout rides out the brief write locks that concurrent workers cause;
    # WAL keeps readers from blocking behind them.
    conn = sqlite3.connect(_CACHE_PATH, timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rerank_scores (
            cache_key       TEXT    PRIMARY KEY,
            model           TEXT    NOT NULL,
            relevance_score REAL    NOT NULL
        )
        """
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def _truncate(document: str) -> str:
    return document[:MAX_DOC_CHARS]


def _cache_key(model: str, query: str, document: str) -> str:
    payload = _SEP.join((model, query, document)).encode()
    return hashlib.sha256(payload).hexdigest()


def _lookup(keys: list[str]) -> dict[str, float]:
    if not keys:
        return {}
    with _get_cache() as conn:
        placeholders = ",".join("?" * len(keys))
        rows = conn.execute(
            f"SELECT cache_key, relevance_score FROM rerank_scores "
            f"WHERE cache_key IN ({placeholders})",
            keys,
        ).fetchall()
    return {key: score for key, score in rows}


def _store(items: list[tuple[str, str, float]]) -> None:
    """``items`` is (cache_key, model, relevance_score)."""
    if not items:
        return
    with _get_cache() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO rerank_scores (cache_key, model, relevance_score) "
            "VALUES (?, ?, ?)",
            items,
        )
        conn.commit()


def _call_api(query: str, documents: list[str], model: str, retries: int = 3) -> list[float]:
    """Score each document against the query; returns scores aligned to ``documents``."""
    headers = {
        "Authorization": f"Bearer {secret_management.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),  # score every document, not just a shortlist
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(RERANK_URL, headers=headers, json=body, timeout=60)
            response.raise_for_status()
            results = response.json()["results"]
            scores = [0.0] * len(documents)
            for result in results:
                scores[result["index"]] = float(result["relevance_score"])
            return scores
        except Exception as error:  # noqa: BLE001 -- network/JSON, retried below
            last_error = error
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"rerank request failed after {retries} attempts") from last_error


def rerank_scores(query: str, documents: list[str], model: str = RERANK_MODEL) -> list[float]:
    """Relevance score for each document against ``query``, aligned to ``documents``.

    Cached per (model, query, document); only unseen pairs hit the network.
    """
    truncated = [_truncate(d) for d in documents]
    keys = [_cache_key(model, query, d) for d in truncated]

    cached = _lookup(keys)
    missing = [i for i, key in enumerate(keys) if key not in cached]

    if missing:
        fresh = _call_api(query, [truncated[i] for i in missing], model)
        to_store = []
        for position, doc_index in enumerate(missing):
            score = fresh[position]
            cached[keys[doc_index]] = score
            to_store.append((keys[doc_index], model, score))
        _store(to_store)

    return [cached[key] for key in keys]


def rerank(query: str, documents: list[str], model: str = RERANK_MODEL) -> list[tuple[int, float]]:
    """(original_index, score) pairs, most relevant first, over ``documents``."""
    scores = rerank_scores(query, documents, model)
    order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in order]
