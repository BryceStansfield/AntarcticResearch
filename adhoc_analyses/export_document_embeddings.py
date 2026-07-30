"""Dump the cached Qwen document embeddings to CSV, with the metadata needed to use them elsewhere.

Two files, one per document class:
  * ``working_paper_embeddings.csv`` — every English Working Paper, once per
    (censorship method x embedding space) variant:
      - censorship ``raw``   — the untouched OCR text (as embedded under type ``WorkingPaper``);
      - censorship ``naive`` — country names replaced by "CountryName" (``censor_text``);
      - censorship ``llm``   — the LLM-detected authorship-revealing phrases stripped
        (``llm_censor_text``), read cache-only so this never triggers paid detection;
      - space ``full``       — the embedding as the embedder produced it;
      - space ``orthogonal`` — the same vector with the direct country-signal subspace projected
        out (``CountrySignalProjector``, the same transform ``country_authorship_classifier
        --orthogonalize-country`` trains on).
  * ``measure_embeddings.csv`` — every embedded instrument of the measure corpus (all four
    instrument types: Measure / Decision / Resolution / Recommendation, plus untyped ones), in the
    ``full`` and ``orthogonal`` spaces. Instrument text is never censored, so its censorship column
    is always ``raw``.

One row per (document, censorship, space): the handful of documents that exceed the embedder's
context window are embedded in several segments, and their segment vectors are averaged into a
single document vector (``n_segments`` records how many were pooled, ``segment_uuids`` names them).
That matches how ``adhoc_analyses.measure_wp_topics`` treats multi-segment papers.

Reads the embedding cache only — a variant whose embedding was never generated is skipped and
counted in the coverage report rather than silently zero-filled. In particular whole-document
LLM-censored embeddings only exist for the papers the authorship classifier covers (those authored
by a target country), so that variant is expected to be a subset. Pass ``--embed-missing`` to
generate the gaps live instead (needs OPENROUTER_API_KEY, and costs money).

Usage:
    uv run python -m adhoc_analyses.export_document_embeddings
    uv run python -m adhoc_analyses.export_document_embeddings --only measures --gzip
"""
import argparse
import collections
import csv
import gzip
import pathlib

import numpy as np
import pandas as pd

from embeddings.document_embeddings import (
    get_embeddings_by_uuid, get_wp_ip_embedding_args, get_representation_of_measure,
    measure_id_to_uuid,
)
from embeddings.embed_all_documents import embed_document_set, CENSORED_WORKING_PAPER_TYPE
from embeddings.working_paper_censorship import (
    get_working_paper_paths, censor_text, llm_censor_text_cached, author_for_stem,
)
from working_paper_authorship.country_signal_projection import CountrySignalProjector

OUTPUT_DIR = pathlib.Path("adhoc_analyses/output")
DOCUMENT_SUMMARY = pathlib.Path("data/antarctic-db/processed/document-summary.parquet")
MEASURE_CORPUS = pathlib.Path("data/MeasureCorpusEnriched.csv")
COUNTRY_DIRECTIONS_PATH = pathlib.Path("data/country_signal/direct_country_directions_allwps.npz")

# document_type recorded for any embedding this script has to generate (--embed-missing). The cache
# is keyed on sha256(text) alone, so these only label new rows — identical text hits the existing
# cache whatever type it was stored under. They name the pipeline step each variant came from.
WP_EMBEDDING_TYPES = {
    "raw": "WorkingPaper",                             # embed_all_documents.embed_all
    "naive": CENSORED_WORKING_PAPER_TYPE,              # embed_all_censored_working_papers
    "llm": "WPAuthorClf::llm_censorship::full",        # country_authorship_classifier
}
MEASURE_EMBEDDING_TYPE = "measure"
SPACES = ("full", "orthogonal")
DOCUMENTS_PER_BATCH = 64  # documents held in memory (with their variant texts) at a time

WP_COLUMNS = ["date", "authors", "title", "document_id", "censorship", "space",
              "meeting_name", "n_segments", "segment_uuids", "embedding"]
MEASURE_COLUMNS = ["date", "title", "document_id", "censorship", "space",
                   "instrument_type", "subject", "adoption_year", "n_segments", "segment_uuids",
                   "embedding"]


# ------------------------------------------------------------------------ metadata

def working_paper_metadata() -> dict[str, dict]:
    """Filename stem -> {date, title, meeting_name} for every Working Paper row of the
    document-summary parquet. Papers with several attachment rows collapse to the first."""
    # Only the columns we need: the parquet's payload_json holds the whole scraped page per row and
    # dwarfs everything else.
    df = pd.read_parquet(DOCUMENT_SUMMARY,
                         columns=["party_type", "paper_url", "meeting_year", "paper_name", "meeting_name"])
    df = df[df["party_type"] == "wp"]
    lookup: dict[str, dict] = {}
    for row in df.itertuples():
        if not isinstance(row.paper_url, str):
            continue
        lookup.setdefault(pathlib.Path(row.paper_url).stem, {
            "date": row.meeting_year,
            "title": (row.paper_name or "").strip(),
            "meeting_name": row.meeting_name,
        })
    return lookup


def metadata_for_stem(lookup: dict[str, dict], stem: str) -> dict:
    """Metadata for a WP filename stem — exact match, then the same substring fallback
    ``author_for_stem`` uses for revision-suffix mismatches. Empty when nothing matches."""
    if stem in lookup:
        return lookup[stem]
    return next((meta for s, meta in lookup.items() if s in stem or stem in s),
                {"date": None, "title": None, "meeting_name": None})


# ------------------------------------------------------------------- vector plumbing

def document_vector(uuids: list[str], cache: dict[str, list[float]]) -> np.ndarray | None:
    """A document's vector: its single segment embedding, or the mean over segments for the few
    documents long enough to be split. ``None`` if any segment is missing from the cache — a
    partially-embedded document must not be exported as if it were whole."""
    vectors = [cache.get(uuid) for uuid in uuids]
    if not vectors or any(v is None for v in vectors):
        return None
    return np.asarray(vectors, dtype=np.float32).mean(axis=0)


def in_space(vector: np.ndarray, space: str, projector: CountrySignalProjector) -> np.ndarray:
    if space == "full":
        return vector
    return projector.project(vector[None, :])[0]


def format_vector(vector: np.ndarray, precision: int) -> str:
    """The vector as a JSON-style list of floats, e.g. "[0.0123, -0.0456, ...]"."""
    return "[" + ", ".join(f"{v:.{precision}g}" for v in vector) + "]"


def open_writer(path: pathlib.Path, columns: list[str], use_gzip: bool):
    handle = (gzip.open(path, "wt", newline="", encoding="utf-8") if use_gzip
              else open(path, "w", newline="", encoding="utf-8"))
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    return handle, writer


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fetch_embeddings(units: list[tuple], embed_missing: bool) -> dict[str, list[float]]:
    """Cached vectors for a batch of ``(uuid, document_type, text)`` embedding units. With
    ``embed_missing`` the uncached ones are generated (paid, live) and re-read."""
    cache = get_embeddings_by_uuid([uuid for uuid, _t, _text in units])
    missing = [unit for unit in units if unit[0] not in cache]
    if missing and embed_missing:
        print(f"    embedding {len(missing)} uncached segment(s)...")
        embed_document_set(missing)
        cache.update(get_embeddings_by_uuid([uuid for uuid, _t, _text in missing]))
    return cache


# --------------------------------------------------------------------------- exports

def export_working_papers(path: pathlib.Path, projector: CountrySignalProjector,
                          precision: int, use_gzip: bool, embed_missing: bool) -> dict:
    """Write one row per (working paper, censorship method, space). Returns per-variant counts."""
    metadata = working_paper_metadata()
    paths = get_working_paper_paths()
    print(f"Working papers: {len(paths)} English documents x {len(WP_EMBEDDING_TYPES)} censorship "
          f"methods x {len(SPACES)} spaces")

    exported: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()
    handle, writer = open_writer(path, WP_COLUMNS, use_gzip)
    try:
        for batch_no, batch in enumerate(batched(paths, DOCUMENTS_PER_BATCH), start=1):
            rows, units = [], []
            for wp_path in batch:
                text = wp_path.read_text(encoding="utf-8", errors="ignore")
                author = author_for_stem(wp_path.stem)
                meta = metadata_for_stem(metadata, wp_path.stem)
                variants = {
                    "raw": text,
                    "naive": censor_text(text),
                    # Cache-only: None when the paper has no known author (nothing to censor
                    # against) or when some chunk was never sent to the detector.
                    "llm": llm_censor_text_cached(text, author) if author else None,
                }
                for method, variant_text in variants.items():
                    if variant_text is None:
                        skipped[f"{method} (uncensorable)"] += 1
                        continue
                    segments = get_wp_ip_embedding_args(variant_text, WP_EMBEDDING_TYPES[method])
                    units.extend(segments)
                    rows.append({
                        "date": meta["date"],
                        "authors": author or "",
                        "title": meta["title"],
                        "document_id": wp_path.stem,
                        "censorship": method,
                        "meeting_name": meta["meeting_name"],
                        "uuids": [uuid for uuid, _t, _seg in segments],
                    })

            cache = fetch_embeddings(units, embed_missing)
            for row in rows:
                uuids = row.pop("uuids")
                vector = document_vector(uuids, cache)
                if vector is None:
                    skipped[f"{row['censorship']} (not embedded)"] += 1
                    continue
                for space in SPACES:
                    writer.writerow({
                        **row,
                        "space": space,
                        "n_segments": len(uuids),
                        "segment_uuids": ";".join(uuids),
                        "embedding": format_vector(in_space(vector, space, projector), precision),
                    })
                    exported[f"{row['censorship']} / {space}"] += 1
            print(f"  batch {batch_no}: {sum(exported.values())} rows written")
    finally:
        handle.close()
    return {"exported": exported, "skipped": skipped}


def export_measures(path: pathlib.Path, projector: CountrySignalProjector,
                    precision: int, use_gzip: bool, embed_missing: bool) -> dict:
    """Write one row per (instrument, space) for every instrument with text in the measure corpus."""
    corpus = pd.read_csv(MEASURE_CORPUS)
    instruments = [row for row in corpus.itertuples() if isinstance(row.Content, str) and row.Content.strip()]
    print(f"Measures: {len(instruments)} instruments with text x {len(SPACES)} spaces")

    exported: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()
    handle, writer = open_writer(path, MEASURE_COLUMNS, use_gzip)
    try:
        for batch in batched(instruments, DOCUMENTS_PER_BATCH):
            rows, units = [], []
            for row in batch:
                uuid = measure_id_to_uuid(row.Document_Number)
                units.append((uuid, MEASURE_EMBEDDING_TYPE, get_representation_of_measure(row)))
                rows.append({
                    # ATCM_Year is the meeting that passed the instrument — the comparable anchor to
                    # a working paper's meeting_year. Adoption_Year (entry into force) is kept
                    # alongside it: it adds ratification lag and is empty for instruments that never
                    # entered into effect.
                    "date": None if pd.isna(row.ATCM_Year) else row.ATCM_Year,
                    "title": row.Title,
                    "document_id": uuid,
                    "censorship": "raw",  # instrument text is never censored
                    "instrument_type": "Untyped" if pd.isna(row.Type) else row.Type,
                    "subject": None if pd.isna(row.Subject) else row.Subject,
                    "adoption_year": None if pd.isna(row.Adoption_Year) else row.Adoption_Year,
                    "uuids": [uuid],
                })

            cache = fetch_embeddings(units, embed_missing)
            for row in rows:
                uuids = row.pop("uuids")
                vector = document_vector(uuids, cache)
                if vector is None:
                    skipped["not embedded"] += 1
                    continue
                for space in SPACES:
                    writer.writerow({
                        **row,
                        "space": space,
                        "n_segments": len(uuids),
                        "segment_uuids": ";".join(uuids),
                        "embedding": format_vector(in_space(vector, space, projector), precision),
                    })
                    exported[f"raw / {space}"] += 1
    finally:
        handle.close()
    return {"exported": exported, "skipped": skipped}


def report(name: str, path: pathlib.Path, counts: dict) -> None:
    print(f"\n{name} -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    for variant, n in sorted(counts["exported"].items()):
        print(f"  {variant:26s} {n:6d} rows")
    for reason, n in sorted(counts["skipped"].items()):
        print(f"  skipped: {reason:17s} {n:6d} documents")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["working-papers", "measures"],
                        help="Export just one of the two files (default: both).")
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR)
    parser.add_argument("--precision", type=int, default=6,
                        help="Significant digits per float (default 6; 9 round-trips float32 "
                             "exactly, at ~30%% more disk).")
    parser.add_argument("--gzip", action="store_true",
                        help="Write .csv.gz instead of .csv — the vectors compress well and the "
                             "plain files run to hundreds of MB.")
    parser.add_argument("--embed-missing", action="store_true",
                        help="Generate any variant embedding that is not cached instead of skipping "
                             "it. Makes live (paid) embedding calls; needs OPENROUTER_API_KEY.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    projector = CountrySignalProjector.from_npz(COUNTRY_DIRECTIONS_PATH)
    print(f"Projecting out a rank-{projector.rank} direct country-signal subspace for the "
          f"'orthogonal' rows ({COUNTRY_DIRECTIONS_PATH}).")
    suffix = ".csv.gz" if args.gzip else ".csv"

    if args.only != "measures":
        path = args.output_dir / f"working_paper_embeddings{suffix}"
        counts = export_working_papers(path, projector, args.precision, args.gzip, args.embed_missing)
        report("Working papers", path, counts)

    if args.only != "working-papers":
        path = args.output_dir / f"measure_embeddings{suffix}"
        counts = export_measures(path, projector, args.precision, args.gzip, args.embed_missing)
        report("Measures", path, counts)


if __name__ == "__main__":
    main()
