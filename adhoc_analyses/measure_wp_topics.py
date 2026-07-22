"""BERTopic over the combined space of ATCM instruments and Working Papers.

Fits a single topic model across both document classes so that a topic is a
shared concept rather than a per-corpus artefact, then reports the
instrument/Working-Paper composition of every topic.

Like ``antarctic_ladder_metrics/topic_introduction.py``, this runs on the Qwen
embeddings already cached in ``data/document_embeddings.sqlite3`` -- neither
re-requests them, since ``embed_all_documents.py`` populates that table up front.
Both are deterministic given a fixed embedding table (UMAP is seeded).

The difference is how the vectors are fetched. This script reads them straight
out of sqlite by document type and pools a document's segments itself, so it
never touches the network and needs no API key at all. ``topic_introduction.py``
hands raw text to BERTopic and lets ``OpenRouterBackend`` resolve it, which means
re-tokenising long documents to recover their segment keys, and an API call for
anything not already cached.

Outputs (to ``adhoc_analyses/output/``):
  * ``combined_topic_breakdown.csv``  — one row per topic, instrument/WP counts
  * ``combined_topic_assignments.csv`` — one row per document, its topic
  * ``combined_topic_words.txt``       — top words per topic, for eyeballing
"""

import collections
import pathlib

import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap import UMAP

import embeddings.document_embeddings as document_embeddings
from embeddings.bertopic_backend import mean_pool, topic_vectorizer

OUTPUT_DIR = pathlib.Path("adhoc_analyses/output")
MEASURE_CORPUS = pathlib.Path("data/MeasureCorpusEnriched.csv")

# Matches topic_introduction.py so the two models stay comparable.
UMAP_RANDOM_STATE = 42
MIN_TOPIC_SIZE = 5


def load_measures() -> list[dict]:
    """Every embedded ATCM instrument (Measure/Recommendation/Resolution/Decision).

    The embedding table stores all four under document_type "measure"; the
    instrument's own ``Type`` is joined back on so topics can be sliced by it.
    """
    getter = document_embeddings.DocumentTextGetter()
    corpus = pd.read_csv(MEASURE_CORPUS).set_index("Document_Number")

    docs = []
    for uuid, embedding in document_embeddings.get_embeddings_by_type("measure"):
        measure_id = int(uuid.removeprefix("MEASURE__"))
        representation = getter.get_measure_representation(measure_id)
        row = corpus.loc[measure_id]
        docs.append(
            {
                "uuid": uuid,
                "doc_class": "Instrument",
                # ~41 embedded instruments carry no Type in the scraped corpus
                # (rules of procedure, consultative-party recognitions, CCAS
                # documents). They are real instruments, so they stay in, tagged
                # "Untyped" rather than silently dropping out of the type counts.
                "instrument_type": "Untyped" if pd.isna(row["Type"]) else row["Type"],
                "text": representation["text"],
                # ATCM_Year (the meeting that passed it) is the comparable
                # anchor to a working paper's meeting_year. Adoption_Year is
                # entry-into-force, which adds ratification lag and is missing
                # for the ~52 instruments that never entered into effect.
                "year": row["ATCM_Year"],
                "adoption_year": row["Adoption_Year"],
                "embedding": np.asarray(embedding, dtype="float32"),
                "label": f"{row['Type']} {measure_id}",
            }
        )
    return docs


def load_working_papers() -> list[dict]:
    """Every embedded English Working Paper, one row per paper (not per segment).

    Long papers are embedded as several segments keyed on sha256(segment), all
    of which map back to the same source file. Those are collapsed to a single
    document with the mean of its segment embeddings.
    """
    getter = document_embeddings.DocumentTextGetter()

    by_file: dict[str, list] = collections.defaultdict(list)
    for uuid, embedding in document_embeddings.get_embeddings_by_type("WorkingPaper"):
        by_file[getter.wp_ip_map[uuid]].append((uuid, embedding))

    docs = []
    for path, segments in by_file.items():
        # Every segment of a file resolves to the same representation (the
        # representation reads the whole file), so the first uuid is enough.
        representation = getter.get_wp_ip_representation(segments[0][0])
        # Papers with no matching row in document-summary.parquet come back with
        # only "text" — no language, so they can't be language-filtered and are
        # dropped, matching topic_introduction.py.
        if str(representation.get("paper_language", "")).lower() != "english":
            continue

        # Same pooling rule the embedding backend uses, so a document's vector
        # is identical whether it comes from here or from OpenRouterBackend.
        mean_embedding = mean_pool([e for _, e in segments])
        docs.append(
            {
                "uuid": segments[0][0],
                "doc_class": "WorkingPaper",
                "instrument_type": "WorkingPaper",
                "text": representation["text"],
                "year": representation["year"],
                "adoption_year": None,
                "embedding": mean_embedding,
                "label": pathlib.Path(path).stem,
                "n_segments": len(segments),
                # Unused here; measure_wp_latency.py needs it to skip papers
                # authored only by non-party bodies when matching.
                "parties": representation.get("parties", []),
            }
        )
    return docs


def fit_combined_topic_model(docs: list[dict]) -> tuple[BERTopic, list[int]]:
    embeddings = np.vstack([d["embedding"] for d in docs])
    texts = [d["text"] for d in docs]

    # Working papers are full OCR'd documents while instruments are a couple of
    # paragraphs, so the c-TF-IDF vocabulary is trimmed to stop the long-document
    # boilerplate from owning every topic representation.
    vectorizer_model = topic_vectorizer()

    # Clustering runs on the precomputed Qwen embeddings passed to fit_transform.
    # Labels come from c-TF-IDF, which downweights corpus-wide vocabulary and so
    # yields distinctive labels here. Embedding-based representation (MMR /
    # KeyBERTInspired) was tried and rejected: it scores candidate words by
    # similarity to the topic centroid, and in a corpus where every document
    # concerns Antarctica that promotes the generic terms -- "antarctic" landed
    # in the top-10 of 112 of 164 topics.
    #
    # With no representation model there is nothing to embed at labelling time,
    # so BERTopic needs no embedding_model at all.
    topic_model = BERTopic(
        umap_model=UMAP(random_state=UMAP_RANDOM_STATE),
        min_topic_size=MIN_TOPIC_SIZE,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    return topic_model, topics


def build_breakdown(topic_model: BERTopic, topics: list[int], docs: list[dict]) -> pd.DataFrame:
    assignments = pd.DataFrame(
        [
            {
                "topic": topic,
                "doc_class": d["doc_class"],
                "instrument_type": d["instrument_type"],
                "year": d["year"],
                "adoption_year": d["adoption_year"],
                "label": d["label"],
                "uuid": d["uuid"],
            }
            for topic, d in zip(topics, docs)
        ]
    )

    instrument_types = ["Measure", "Recommendation", "Resolution", "Decision", "Untyped"]
    rows = []
    for topic, group in assignments.groupby("topic"):
        n_wp = int((group["doc_class"] == "WorkingPaper").sum())
        n_instrument = int((group["doc_class"] == "Instrument").sum())
        words = topic_model.get_topic(topic)
        rows.append(
            {
                "topic": topic,
                "size": len(group),
                "n_working_papers": n_wp,
                "n_instruments": n_instrument,
                "instrument_share": n_instrument / len(group),
                **{
                    f"n_{t.lower()}": int((group["instrument_type"] == t).sum())
                    for t in instrument_types
                },
                "first_year": _as_year(group["year"].min()),
                "last_year": _as_year(group["year"].max()),
                "first_wp_year": _min_year(group, "WorkingPaper"),
                "first_instrument_year": _min_year(group, "Instrument"),
                "top_words": ", ".join(w for w, _ in words) if isinstance(words, list) else "",
            }
        )

    breakdown = pd.DataFrame(rows).sort_values(
        ["instrument_share", "size"], ascending=[False, False]
    )
    return breakdown, assignments


def _as_year(value):
    """Years are nullable — one instrument has no ATCM_Year, and a handful of
    working papers have no matching row in document-summary.parquet."""
    return None if pd.isna(value) else int(value)


def _min_year(group: pd.DataFrame, doc_class: str):
    years = group.loc[group["doc_class"] == doc_class, "year"].dropna()
    return _as_year(years.min()) if len(years) else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached embeddings...")
    measures = load_measures()
    working_papers = load_working_papers()
    docs = measures + working_papers
    print(f"  {len(measures)} instruments, {len(working_papers)} English working papers")

    topic_model, topics = fit_combined_topic_model(docs)
    breakdown, assignments = build_breakdown(topic_model, topics, docs)

    breakdown.to_csv(OUTPUT_DIR / "combined_topic_breakdown.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "combined_topic_assignments.csv", index=False)

    with open(OUTPUT_DIR / "combined_topic_words.txt", "w") as f:
        for topic in sorted(t for t in set(topics)):
            words = topic_model.get_topic(topic)
            if isinstance(words, list):
                f.write(f"Topic {topic}: " + ", ".join(f"{w}({s:.3f})" for w, s in words) + "\n")

    real = breakdown[breakdown["topic"] != -1]
    outliers = breakdown.loc[breakdown["topic"] == -1, "size"]
    n_outlier = int(outliers.iloc[0]) if len(outliers) else 0

    print(f"\n{len(real)} topics + {n_outlier} outlier documents ({n_outlier / len(docs):.1%} of corpus)")

    # The headline: is a topic a shared concept, or just a document class?
    buckets = {
        "pure working paper (0 instruments)": real["n_instruments"] == 0,
        "WP-dominated (<25% instrument)": (real["instrument_share"] > 0) & (real["instrument_share"] < 0.25),
        "mixed (25-75% instrument)": (real["instrument_share"] >= 0.25) & (real["instrument_share"] <= 0.75),
        "instrument-dominated (>75%)": (real["instrument_share"] > 0.75) & (real["n_working_papers"] > 0),
        "pure instrument (0 WPs)": real["n_working_papers"] == 0,
    }
    print("\nTopic composition:")
    for name, mask in buckets.items():
        print(f"  {name:<36} {int(mask.sum()):>4} topics  {int(real.loc[mask, 'size'].sum()):>5} docs")

    print("\nMost instrument-heavy topics:")
    print(real.head(15).to_string(index=False, max_colwidth=55))
    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
