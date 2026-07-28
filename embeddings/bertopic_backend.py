"""A BERTopic embedding backend backed by the cached OpenRouter embeddings.

Why this exists
---------------
BERTopic resolves whatever you hand to ``embedding_model=`` through
``bertopic.backend.select_backend``. That function recognises a backend either by
``isinstance(..., BaseEmbedder)`` or by string-matching the type name against a
fixed list ("sentence_transformers", "flair", "spacy", ...). A duck-typed object
that merely exposes ``.encode`` matches *nothing* and falls through to the final
line of ``select_backend``, which silently returns
``SentenceTransformerBackend("all-MiniLM-L6-v2")``.

So a plain class is not just ignored for word embeddings — it is replaced
wholesale, and the model quietly runs on 384-dim MiniLM with a 256-token input
limit instead of the intended Qwen3-8B embeddings.

Subclassing ``BaseEmbedder`` is what makes ``select_backend`` return the backend
untouched.

Full documents, not first segments
----------------------------------
Documents longer than the embedding model's context window are split by
``split_long_document``. This backend embeds *every* segment and mean-pools them
into one vector for the document, so a long working paper is represented by all
of its text. Taking ``get_wp_ip_embedding_args(...)[0]`` instead — the first
segment only — silently discards everything past the first ~32k tokens.

Source vectors are unit-norm, so the pooled vector is re-normalised to keep it
on the same scale as a single-segment document.

Caching
-------
Segments are keyed the same way as the rest of the pipeline — ``sha256`` of the
segment from ``get_wp_ip_embedding_args`` — so documents that are already
embedded (every working paper, every information paper) are free cache hits.
Lookups ignore ``document_type``, so a hit is a hit regardless of which run first
stored it.

Newly generated vectors are written under this backend's own ``document_type``
(default ``BERTopicRepr``) rather than a corpus type. Topic representation embeds
*individual words*, and writing those under "WorkingPaper" would corrupt
``get_embeddings_by_type("WorkingPaper")`` for every downstream consumer.
"""

import concurrent.futures
import hashlib

import numpy as np
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

import embeddings.document_embeddings as document_embeddings

REPRESENTATION_TYPE = "BERTopicRepr"

# Seed for the shared UMAP below. BERTopic leaves UMAP unseeded, which makes a fit
# irreproducible; every topic model here fixes it to the same value.
UMAP_RANDOM_STATE = 42

# ATCM documents carry a masthead repeating the meeting's name in English,
# Spanish, French and Russian ("ANTARCTIC TREATY / TRATADO ANTARTICO / FOURTH
# CONSULTATIVE MEETING / CUARTA REUNION CONSULTIVA / QUATRIEME REUNION
# CONSULTATIVE / TRAITE SUR L'ANTARCTIQUE"). In short early papers that
# boilerplate is a large fraction of the text, so its tokens dominate the
# c-TF-IDF labels of any cluster of same-meeting documents.
#
# Only the non-English tokens are listed. English masthead words ("antarctic",
# "treaty", "meeting", "consultative") are left in: they are ordinary content
# words elsewhere in the corpus, and dropping them would blind the labels to
# real topics. This affects labels only -- clustering runs on the embeddings.
MASTHEAD_STOP_WORDS = frozenset({
    "tratado", "antartico", "cuarta", "reunion", "consultiva",
    "traite", "quatrieme", "antarctique",
    "la", "las", "los", "el", "les", "del", "sur", "en", "du", "des",
})


def bertopic_umap(random_state: int = UMAP_RANDOM_STATE):
    """BERTopic's own default dimensionality reducer, seeded so a fit is reproducible.

    Read off an unfitted BERTopic rather than constructed here, because the two sets of
    defaults are not the same reducer at all. BERTopic builds
    ``UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")`` when
    ``umap_model`` is left unset -- five dimensions, clusters allowed to compact fully, and
    cosine, all chosen so HDBSCAN has room to separate them. A bare ``UMAP(random_state=42)``
    is *umap-learn's* defaults instead: two components, ``min_dist=0.1``, euclidean. Passing
    one to seed the fit therefore silently swapped the clustering space for a far more
    crowded one, and everything HDBSCAN could not fit into it became an outlier.

    Taking the instance BERTopic constructs keeps these models on BERTopic's defaults if
    those ever move; the seed is the only thing overridden.
    """
    umap_model = BERTopic().umap_model
    umap_model.random_state = random_state
    return umap_model


def topic_vectorizer(min_df: int = 2) -> CountVectorizer:
    """Shared c-TF-IDF vectoriser, so the working-paper model and the combined
    working-paper + instrument model label topics on the same basis."""
    return CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | MASTHEAD_STOP_WORDS),
        min_df=min_df,
    )

# SQLite caps host parameters per statement; stay well under it.
_SQL_CHUNK = 900


def segment_keys(text: str) -> list[str]:
    """Cache keys for every segment of a document, in order."""
    return [
        hashlib.sha256(segment.encode()).hexdigest()
        for segment in document_embeddings.split_long_document(text)
    ]


def mean_pool(vectors) -> np.ndarray:
    """Combine a document's segment embeddings into one unit-norm vector.

    Inputs are unit-norm, so a plain mean shrinks toward the origin as segments
    disagree; re-normalising keeps every document comparable regardless of how
    many segments it was split into.
    """
    pooled = np.mean(np.asarray(vectors, dtype="float32"), axis=0)
    norm = np.linalg.norm(pooled)
    return pooled if norm == 0 else pooled / norm


def _lookup_many(keys: list[str], model_uuid: str) -> dict[str, list[float]]:
    """One query per chunk instead of one per key — representation embeds
    thousands of short words, and per-key queries dominate the runtime."""
    found = {}
    with document_embeddings.get_connection() as conn:
        for start in range(0, len(keys), _SQL_CHUNK):
            chunk = keys[start : start + _SQL_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT document_uuid, embedding FROM embeddings "
                f"WHERE model_uuid=? AND document_uuid IN ({placeholders})",
                (model_uuid, *chunk),
            ).fetchall()
            for uuid, blob in rows:
                found[uuid] = document_embeddings.array.array("f", blob).tolist()
    return found


class OpenRouterBackend(BaseEmbedder):
    """BERTopic backend over OpenRouter embeddings, cached in the shared sqlite DB."""

    def __init__(
        self,
        document_type: str = REPRESENTATION_TYPE,
        model_uuid: str = document_embeddings.DEFAULT_EMBEDDING_MODEL,
        max_workers: int = 64,
    ) -> None:
        super().__init__()
        self.document_type = document_type
        self.model_uuid = model_uuid
        self.max_workers = max_workers

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        # Split once — split_long_document tokenizes, which is far too slow to
        # repeat per document.
        segmented = [document_embeddings.split_long_document(d) for d in documents]
        per_document = [
            [hashlib.sha256(s.encode()).hexdigest() for s in segments]
            for segments in segmented
        ]

        # Dedupe: topic representation asks for the same word across many topics.
        unique: dict[str, str] = {}
        for segments, keys in zip(segmented, per_document):
            for key, segment in zip(keys, segments):
                unique.setdefault(key, segment)

        cached = _lookup_many(list(unique), self.model_uuid)
        missing = {k: v for k, v in unique.items() if k not in cached}

        if missing:
            if verbose:
                print(
                    f"  OpenRouterBackend: {len(cached)} cached, "
                    f"embedding {len(missing)} new of {len(unique)} unique segments"
                )
            cached.update(self._generate(missing))

        return np.array(
            [mean_pool([cached[k] for k in keys]) for keys in per_document],
            dtype="float32",
        )

    def _generate(self, missing: dict[str, str]) -> dict[str, list[float]]:
        """Embeddings are I/O bound, so fan out over threads. Each worker writes
        its own row through ``generate_embedding`` (which caches as it goes)."""

        def one(item):
            key, text = item
            for attempt in range(3):
                try:
                    return key, document_embeddings.generate_embedding(
                        key, self.document_type, text, self.model_uuid
                    )
                except Exception:
                    if attempt == 2:
                        raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return dict(pool.map(one, missing.items()))
