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

import argparse
import pathlib

import numpy as np
import pandas as pd
from bertopic import BERTopic

import embeddings.document_embeddings as document_embeddings
from embeddings.bertopic_backend import bertopic_umap, topic_vectorizer
from working_paper_authorship.country_signal_projection import CountrySignalProjector

OUTPUT_DIR = pathlib.Path("adhoc_analyses/output")
MEASURE_CORPUS = pathlib.Path("data/MeasureCorpusEnriched.csv")
# Direct country-signal directions recovered by direct_country_signal_probe --all-wps.
COUNTRY_DIRECTIONS_PATH = pathlib.Path("data/country_signal/direct_country_directions_allwps.npz")

# Matches topic_introduction.py so the two models stay comparable: both cluster with
# bertopic_umap() (BERTopic's own default reducer, seeded) at this minimum topic size.
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
        # ~41 embedded instruments carry no Type in the scraped corpus (rules of
        # procedure, consultative-party recognitions, CCAS documents). They are
        # real instruments, so they stay in, tagged "Untyped" rather than
        # silently dropping out of the type counts. Resolved once and used for
        # the label too: interpolating the raw cell rendered a missing Type as
        # the literal string "nan", so those instruments appeared on every
        # figure that shows a label as "nan 123".
        instrument_type = "Untyped" if pd.isna(row["Type"]) else row["Type"]
        docs.append(
            {
                "uuid": uuid,
                "doc_class": "Instrument",
                "instrument_type": instrument_type,
                "text": representation["text"],
                # ATCM_Year (the meeting that passed it) is the comparable
                # anchor to a working paper's meeting_year. Adoption_Year is
                # entry-into-force, which adds ratification lag and is missing
                # for the ~52 instruments that never entered into effect.
                "year": row["ATCM_Year"],
                "adoption_year": row["Adoption_Year"],
                "embedding": np.asarray(embedding, dtype="float32"),
                "label": f"{instrument_type} {measure_id}",
            }
        )
    return docs


def load_working_papers() -> list[dict]:
    """Every embedded English Working Paper, one row per paper (not per segment).

    The grouping and pooling that turns per-segment embedding rows back into
    per-document vectors lives in ``DocumentTextGetter.get_all_of_type`` -- it is
    needed by every consumer of the WorkingPaper type, not just this one, and
    keeping a private copy here is what let the other consumers drift into
    counting segments as documents. This only reshapes its output into the
    document dicts the rest of this module (and ``measure_wp_latency``) expects.

    Papers with no matching row in document-summary.parquet come back with only
    "text" -- no language, so they can't be language-filtered and are dropped.
    """
    docs = []
    for d in document_embeddings.DocumentTextGetter().get_all_of_type(
        "WorkingPaper", with_embeddings=True
    ):
        if str(d.get("paper_language", "")).lower() != "english":
            continue
        docs.append(
            {
                "uuid": d["uuid"],
                "doc_class": "WorkingPaper",
                "instrument_type": "WorkingPaper",
                "text": d["text"],
                "year": d["year"],
                "adoption_year": None,
                "embedding": d["embedding"],
                "label": pathlib.Path(d["source"]).stem,
                "n_segments": d["n_segments"],
                # Unused here; measure_wp_latency.py needs it to skip papers
                # authored only by non-party bodies when matching.
                "parties": d.get("parties", []),
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
        umap_model=bertopic_umap(),
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


def orthogonalize_embeddings(docs: list[dict]) -> int:
    """Project the direct country-signal subspace out of every document's embedding, in place.

    Same operator the authorship classifier's --orthogonalize-country uses. Projection is linear so it
    commutes with the WP segment mean-pooling already done in load_working_papers. Returns the rank
    removed. NOTE: UMAP+HDBSCAN is highly sensitive to dropping a whole subspace, so the resulting
    clusters are not directly comparable to the baseline — this only shows what the topics look like."""
    projector = CountrySignalProjector.from_npz(COUNTRY_DIRECTIONS_PATH)
    stacked = np.vstack([d["embedding"] for d in docs])
    projected = projector.transform(stacked)
    for d, e in zip(docs, projected):
        d["embedding"] = e
    return projector.rank


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--orthogonalize-country", action="store_true",
        help="Project the direct country-signal subspace (from direct_country_signal_probe) out of the "
             "embeddings before clustering, to see what the topics look like without the direct signal. "
             "Writes *_orthogonal output files so the baseline topics are not overwritten. Caveat: "
             "UMAP+HDBSCAN is sensitive to dropping dimensions, so clusters shift for that reason alone.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached embeddings...")
    measures = load_measures()
    working_papers = load_working_papers()
    docs = measures + working_papers
    print(f"  {len(measures)} instruments, {len(working_papers)} English working papers")

    suffix = ""
    if args.orthogonalize_country:
        rank = orthogonalize_embeddings(docs)
        suffix = "_orthogonal"
        print(f"  projected out the rank-{rank} direct country-signal subspace before clustering")

    topic_model, topics = fit_combined_topic_model(docs)
    breakdown, assignments = build_breakdown(topic_model, topics, docs)

    breakdown.to_csv(OUTPUT_DIR / f"combined_topic_breakdown{suffix}.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / f"combined_topic_assignments{suffix}.csv", index=False)

    with open(OUTPUT_DIR / f"combined_topic_words{suffix}.txt", "w") as f:
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


from utils import line_buffer_stdout

if __name__ == "__main__":
    line_buffer_stdout()
    main()
