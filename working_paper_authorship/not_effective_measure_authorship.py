"""Predict authorship of the not-yet-effective measures with the pretrained WP authorship models.

Takes the measures in ``data/Not-Effective measures.csv`` and scores each one with the working-paper
authorship classifiers trained by ``country_authorship_classifier`` — specifically the models trained
on *censored* documents (``naive__full`` and ``llm_censorship__full``). The measure text itself is
**not** censored: we embed the raw ``Content`` once (whole-document, "full" granularity, same embedder
and hashing the WP pipeline uses) and feed those embeddings straight into the censored-trained models.
This deliberately mirrors "train on censored WPs, predict on uncensored measures".

Ground truth comes from the CSV's ``TRUE`` / ``FALSE`` columns: a target country listed in ``TRUE`` is
treated as an author (label 1), one listed in ``FALSE`` as a non-author (label 0). Only the five target
countries the models were trained on are scored (Australia, United Kingdom, United States, Norway,
Chile); every measure lists all five across TRUE/FALSE, so no country is dropped.

Outputs, per (model, censored-dataset) pair:
  * a per-measure predictions CSV in data/not_effective_measure_predictions/, and
  * a combined validation-style report (cross-entropy, exact match, per-country recall/precision)
    at data/not_effective_measure_predictions/report.txt.

The pretrained model pickles must already exist under data/author_classification_models/ (run
country_authorship_classifier first). This script never trains.
"""
import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd

from working_paper_authorship import country_authorship_classifier as cc
from embeddings.document_embeddings import get_embedding, has_embedding
from embeddings.embed_all_documents import embed_document_set

MEASURES_CSV = pathlib.Path("data/Not-Effective measures.csv")
OUTPUT_DIR = pathlib.Path("data/not_effective_measure_predictions")
ORTHOGONAL_OUTPUT_DIR = pathlib.Path("data/not_effective_measure_predictions_orthogonal")

# The censored-trained datasets we score the (uncensored) measures with. Input text is never
# censored — only the *models* were trained on censored working papers.
CENSORED_DATASETS = ["naive__full", "llm_censorship__full"]


def parse_countries(cell) -> set[str]:
    """Comma-separated country cell -> the subset of target COUNTRIES it names (canonicalised)."""
    if pd.isna(cell):
        return set()
    out = set()
    for raw in str(cell).split(","):
        name = raw.strip()
        if name in cc._alias_to_canonical:
            out.add(cc._alias_to_canonical[name])
    return out


def load_measure_records() -> list[dict]:
    """One record per measure as the classifier's {stem, text, label, author}.

    ``text`` is the raw (uncensored) measure Content; ``label`` is the 5-country author vector from
    the TRUE/FALSE columns. Rows with empty content are skipped."""
    df = pd.read_csv(MEASURES_CSV)
    records = []
    for row in df.itertuples():
        content = getattr(row, "Content", None)
        if pd.isna(content) or not str(content).strip():
            continue
        authors = parse_countries(row.TRUE)
        non_authors = parse_countries(getattr(row, "FALSE"))
        missing = [c for c in cc.COUNTRIES if c not in authors and c not in non_authors]
        if missing:
            raise ValueError(
                f"Measure {row.Document_Number}: target countries {missing} appear in neither "
                f"TRUE nor FALSE — cannot assign a ground-truth label."
            )
        records.append({
            "stem": str(row.Document_Number),
            "text": str(content),
            "label": np.array([1 if c in authors else 0 for c in cc.COUNTRIES], dtype=np.int32),
            "author": "",  # unused: raw (uncensored) input, so no LLM-censor author hint needed
        })
    return records


def embed_measures(records: list[dict]) -> list[tuple]:
    """Chunk each measure whole-document ("full", uncensored), embed any uncached segments, and
    return the (hash, label, stem) plan linking every embedded row back to its measure."""
    embed_units, hash_labels = cc.dataset_units(records, method="raw", granularity="full")
    unique = {u[0]: u for u in embed_units}
    print(f"Embedding {len(unique)} unique measure segments (cached ones are skipped)...")
    embed_document_set(list(unique.values()))

    missing = [h for h in unique if not has_embedding(h)]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(unique)} measure segments still have no embedding after "
            f"embed_document_set — aborting rather than scoring a partial set."
        )
    return hash_labels


def pooled_predictions(model, hash_labels: list[tuple]) -> pd.DataFrame:
    """Score every measure segment, then pool to one row per measure.

    A long measure splits into several segments; its probabilities are averaged over them and the
    prediction re-derived by thresholding the mean at 0.5 (``n_units`` = segments pooled). Short
    measures are a single segment. Returns a frame with, per measure: true/pred/prob per country."""
    X, Y, stems = cc.assemble_xy(hash_labels)
    proba = cc.positive_proba(model, X)

    df = pd.DataFrame({"stem": stems})
    for i, country in enumerate(cc.COUNTRIES):
        df[f"true__{country}"] = Y[:, i]
        df[f"prob__{country}"] = proba[:, i]
    df["n_units"] = 1

    prob_cols = [f"prob__{c}" for c in cc.COUNTRIES]
    agg = {**{f"true__{c}": "first" for c in cc.COUNTRIES},
           **{c: "mean" for c in prob_cols}, "n_units": "sum"}
    df = df.groupby("stem", as_index=False, sort=True).agg(agg)
    for country in cc.COUNTRIES:
        df[f"pred__{country}"] = (df[f"prob__{country}"] >= 0.5).astype(int)
    return df


def metrics_from_pooled(df: pd.DataFrame) -> dict:
    """Per-country recall/precision, exact-match and mean cross-entropy from a pooled frame."""
    Y_true = np.column_stack([df[f"true__{c}"].to_numpy() for c in cc.COUNTRIES]).astype(int)
    Y_pred = np.column_stack([df[f"pred__{c}"].to_numpy() for c in cc.COUNTRIES]).astype(int)
    Y_prob = np.column_stack([df[f"prob__{c}"].to_numpy() for c in cc.COUNTRIES]).astype(float)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    return {
        "per_country_recall": [float(recall_score(Y_true[:, i], Y_pred[:, i], zero_division=0)) for i in range(len(cc.COUNTRIES))],
        "per_country_precision": [float(precision_score(Y_true[:, i], Y_pred[:, i], zero_division=0)) for i in range(len(cc.COUNTRIES))],
        "exact": float(accuracy_score(Y_true, Y_pred)),
        "loss": cc.mean_cross_entropy(Y_true, Y_prob),
    }


def write_predictions_csv(df: pd.DataFrame, model_name: str, slug: str) -> pathlib.Path:
    columns = ["stem", "n_units"] + [f"{p}__{c}" for c in cc.COUNTRIES for p in ("true", "pred", "prob")]
    path = OUTPUT_DIR / f"{cc.model_slug(model_name)}__{slug}.csv"
    df[columns].to_csv(path, index=False)
    return path


def random_guess_baseline(records: list[dict]) -> tuple[float, list[float]]:
    """No-skill BCE baseline: predict each country's base rate over these measures."""
    from sklearn.metrics import log_loss
    labels = np.array([r["label"] for r in records])
    base_rates = labels.mean(axis=0)
    proba = np.tile(base_rates, (len(labels), 1))
    per_class = [float(log_loss(labels[:, i], proba[:, i], labels=[0, 1])) for i in range(len(cc.COUNTRIES))]
    return float(np.mean(per_class)), per_class


def write_report(results: list[dict], n_measures: int, baseline_avg: float, baseline_per_class: list[float],
                 orthogonalized: bool = False) -> None:
    baseline_cols = " ".join(f"{b:.2f}" for b in baseline_per_class)
    ortho_note = (" (COUNTRY-ORTHOGONALIZED: direct-signal subspace projected out of both the training "
                  "embeddings and these measure embeddings)") if orthogonalized else ""
    lines = [
        f"NOT-YET-EFFECTIVE MEASURES — AUTHORSHIP BY CENSORED-TRAINED WP MODELS (uncensored input){ortho_note}",
        f"Countries: {', '.join(cc.COUNTRIES)}",
        f"Measures scored: {n_measures}  (raw measure Content, embedded whole-document, no censorship)",
        "Models were trained on CENSORED working papers; the measure text fed to them is NOT censored.",
        f"Random-guess BCE baseline (measure base rates): {baseline_avg:.4f}  per-class[{baseline_cols}]",
        f"Per-country recall / precision order: {', '.join(cc.COUNTRIES)}",
        "",
        f"{'model':20s} {'trained-on dataset':22s} {'x-entropy':>10s} {'exact':>7s}  per-country recall / precision",
    ]
    order = {slug: i for i, slug in enumerate(CENSORED_DATASETS)}
    for r in sorted(results, key=lambda r: (r["model"], order[r["dataset"]])):
        rec = " ".join(f"{x:.2f}" for x in r["per_country_recall"])
        prec = " ".join(f"{p:.2f}" for p in r["per_country_precision"])
        lines.append(f"{r['model']:20s} {r['dataset']:22s} {r['loss']:10.4f} {r['exact']:7.4f}  rec[{rec}] prec[{prec}]")
    report = "\n".join(lines)
    (OUTPUT_DIR / "report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote report + {len(results)} prediction CSVs to {OUTPUT_DIR}/")


def run(orthogonalize_country: bool = False) -> list[dict]:
    global OUTPUT_DIR
    model_dir = cc.OUTPUT_DIR
    if orthogonalize_country:
        # Project the direct country-signal subspace out of the measure embeddings (via cc.assemble_xy)
        # and score with the models trained on the same orthogonalized space — the OOD analogue of the
        # --orthogonalize-country WP benchmark.
        cc._PROJECTOR = cc.CountrySignalProjector.from_npz(cc.COUNTRY_DIRECTIONS_PATH)
        model_dir = cc.ORTHOGONAL_OUTPUT_DIR
        OUTPUT_DIR = ORTHOGONAL_OUTPUT_DIR
        print(f"Orthogonalizing measure embeddings (rank-{cc._PROJECTOR.rank}) and scoring with orthogonal "
              f"models from {model_dir}/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading not-yet-effective measures...")
    records = load_measure_records()
    print(f"  measures: {len(records)}")

    hash_labels = embed_measures(records)

    results = []
    for slug in CENSORED_DATASETS:
        for name in cc.MODEL_NAMES:
            pickle_path = model_dir / f"{cc.model_slug(name)}__{slug}.pickle"
            if not pickle_path.exists():
                print(f"  [skip] no pretrained model at {pickle_path}")
                continue
            with open(pickle_path, "rb") as f:
                model = pickle.load(f)

            df = pooled_predictions(model, hash_labels)
            metrics = metrics_from_pooled(df)
            pred_path = write_predictions_csv(df, name, slug)
            results.append({"model": name, "dataset": slug, **metrics})
            print(f"  {name:20s} on {slug:22s} loss={metrics['loss']:.4f} exact={metrics['exact']:.4f}"
                  f"  -> {pred_path.name}")

    baseline_avg, baseline_per_class = random_guess_baseline(records)
    write_report(results, len(records), baseline_avg, baseline_per_class, orthogonalized=orthogonalize_country)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--orthogonalize-country", action="store_true",
        help="Project the direct country-signal subspace out of the measure embeddings and score with "
             "the country-orthogonal models (data/author_classification_models_orthogonal/). Writes to "
             "data/not_effective_measure_predictions_orthogonal/.",
    )
    args = parser.parse_args()
    run(orthogonalize_country=args.orthogonalize_country)


if __name__ == "__main__":
    main()
