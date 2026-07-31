"""Does squeezing the PCA down to very few dimensions curb the classifiers' reliance on non-topic
(authorial-fingerprint) directions that predict authorship?

Hypothesis: the whole-document authorship classifiers may be keying on fine, non-topic directions in
the embedding (stylistic / boilerplate fingerprints of an authoring party) rather than on topic. If
so, keeping only the top few PCA components — which carry the coarse, mostly-topical variance —
should hold up on the WP validation set only insofar as authorship is topic-correlated, and should
change how the models transfer to the not-yet-effective measures.

This trains, for each whole-document dataset (raw / naive / llm_censorship) and each model, a family
of pipelines with PCA fixed to {2, 4, 8} components. Every other hyperparameter is held at that
dataset's already-tuned value (from best_hyperparameters__*.json) so the ONLY thing varying is the
PCA width. It then reports, side by side with the existing full-width model:
  1) validation-set performance (WP held-out val), and
  2) transfer performance on the 52 measures (raw uncensored Content, same as
     not_effective_measure_authorship — the measure text is never censored).

Nothing here overwrites the production benchmark: low-dim models go to a separate directory with a
``pcaN`` suffix, and the full-width models are only ever read.
"""
import json
import pathlib
import pickle

import numpy as np

from working_paper_authorship import country_authorship_classifier as cc
from working_paper_authorship import not_effective_measure_authorship as nema
from embeddings.document_embeddings import has_embedding
from embeddings.embed_all_documents import embed_document_set

PCA_DIMS = [2, 4, 8]
# Whole-document datasets only: they have cached hyperparameters + embeddings, and the measure
# pipeline is whole-document. (slug -> censorship method.)
DATASETS = {
    "raw__full": "raw",
    "naive__full": "naive",
    "llm_censorship__full": "llm_censorship",
}

FULL_MODELS_DIR = cc.OUTPUT_DIR                                      # read-only: the production models
OUTPUT_DIR = pathlib.Path("data/author_classification_models_lowdim")
PREDICTIONS_DIR = pathlib.Path("data/lowdim_measure_predictions")
REPORT_PATH = OUTPUT_DIR / "lowdim_report.txt"


def make_lowdim(name: str, best_params: dict, n_components: int):
    """The dataset's tuned pipeline for ``name`` with PCA forced to ``n_components`` components."""
    pipe = cc.base_pipeline(name)
    params = dict(best_params)
    params["pca__n_components"] = n_components
    pipe.set_params(**params)
    return pipe


def prepare_dataset(method: str, train, val):
    """Embed (cached ones skipped) and assemble (X, Y) for train and val of one dataset."""
    plan, unique = {}, {}
    for split_name, recs in (("train", train), ("val", val)):
        units, hash_labels = cc.dataset_units(recs, method)
        for unit in units:
            unique.setdefault(unit[0], unit)
        plan[split_name] = hash_labels
    embed_document_set(list(unique.values()))
    missing = [h for h in unique if not has_embedding(h)]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(unique)} fragments have no cached embedding for this dataset — "
            f"run country_authorship_classifier first so the WP embeddings are cached."
        )
    X_train, Y_train, _ = cc.assemble_xy(plan["train"])
    X_val, Y_val, _ = cc.assemble_xy(plan["val"])
    return X_train, Y_train, X_val, Y_val


def measure_metrics(model, measure_hash_labels, name: str, slug: str, dim_tag: str) -> dict:
    """Score the measures with ``model``, write the per-measure CSV, return pooled metrics."""
    df = nema.pooled_predictions(model, measure_hash_labels)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["stem", "n_units"] + [f"{p}__{c}" for c in cc.COUNTRIES for p in ("true", "pred", "prob")]
    df[columns].to_csv(PREDICTIONS_DIR / f"{cc.model_slug(name)}__{slug}__{dim_tag}.csv", index=False)
    return nema.metrics_from_pooled(df)


def load_full_model(name: str, slug: str):
    path = FULL_MODELS_DIR / f"{cc.model_slug(name)}__{slug}.pickle"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_lowdim_model(name: str, slug: str, best_params: dict, dim: int, X_train, Y_train):
    """Train (or load a previously trained) low-dim model. Never touches the full-width models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{cc.model_slug(name)}__{slug}__pca{dim}.pickle"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    model = make_lowdim(name, best_params, dim)
    model.fit(X_train, Y_train)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def run() -> list[dict]:
    print("Loading working papers + authors...")
    records = cc.load_working_papers()
    train, val, _test = cc.split_documents(records)
    print(f"  WP docs: {len(records)} (train {len(train)}, val {len(val)}); test held out, unused")

    print("Loading + embedding the not-yet-effective measures (uncensored)...")
    measure_records = nema.load_measure_records()
    measure_hash_labels = nema.embed_measures(measure_records)
    print(f"  measures: {len(measure_records)}")

    rows = []
    for slug, method in DATASETS.items():
        print(f"\n=== dataset {slug} ===")
        X_train, Y_train, X_val, Y_val = prepare_dataset(method, train, val)
        best_params = json.loads((FULL_MODELS_DIR / f"best_hyperparameters__{slug}.json").read_text())
        print(f"  train {X_train.shape}, val {X_val.shape}")

        for name in cc.MODEL_NAMES:
            if name not in best_params:
                continue
            for dim in ["full", *PCA_DIMS]:
                if dim == "full":
                    model = load_full_model(name, slug)
                    if model is None:
                        continue
                    dim_tag = "full"
                else:
                    model = get_lowdim_model(name, slug, best_params[name], dim, X_train, Y_train)
                    dim_tag = f"pca{dim}"

                val_m = cc._evaluate(model, X_val, Y_val)
                meas_m = measure_metrics(model, measure_hash_labels, name, slug, dim_tag)
                rows.append({
                    "dataset": slug, "model": name, "dim": dim,
                    "val_loss": val_m["loss"], "val_exact": val_m["exact"],
                    "meas_loss": meas_m["loss"], "meas_exact": meas_m["exact"],
                    "meas_recall": meas_m["per_country_recall"],
                    "val_recall": val_m["per_country_recall"],
                })
                print(f"  {name:20s} {dim_tag:5s}  val_xent={val_m['loss']:.4f} "
                      f"val_exact={val_m['exact']:.3f}  meas_xent={meas_m['loss']:.4f} "
                      f"meas_exact={meas_m['exact']:.3f}")

    val_baseline, _ = cc.random_guess_baseline(val)
    meas_baseline, _ = nema.random_guess_baseline(measure_records)
    write_report(rows, val_baseline, meas_baseline, len(val), len(measure_records))
    return rows


def write_report(rows, val_baseline, meas_baseline, n_val, n_meas) -> None:
    dim_order = {"full": 0, 2: 1, 4: 2, 8: 3}
    lines = [
        "LOW-DIMENSIONAL PCA EXPERIMENT — does dropping non-topic dimensions curb authorship leakage?",
        f"Countries: {', '.join(cc.COUNTRIES)}",
        "Each dataset's tuned classifier hyperparameters are held fixed; ONLY the PCA width varies.",
        f"Validation = WP held-out val ({n_val} docs).  Measures = {n_meas} not-yet-effective measures",
        "(raw uncensored Content, whole-document, same embeddings as not_effective_measure_authorship).",
        f"Random-guess BCE baselines — validation: {val_baseline:.4f}   measures: {meas_baseline:.4f}  "
        "(lower cross-entropy is better)",
        f"Per-country recall order: {', '.join(cc.COUNTRIES)}",
        "",
        f"{'dataset':20s} {'model':20s} {'dim':5s} {'val_xent':>8s} {'val_ex':>6s} "
        f"{'meas_xent':>9s} {'meas_ex':>7s}  meas recall / val recall",
    ]
    for r in sorted(rows, key=lambda r: (r["dataset"], r["model"], dim_order[r["dim"]])):
        mrec = " ".join(f"{x:.2f}" for x in r["meas_recall"])
        vrec = " ".join(f"{x:.2f}" for x in r["val_recall"])
        dim_tag = "full" if r["dim"] == "full" else f"pca{r['dim']}"
        lines.append(
            f"{r['dataset']:20s} {r['model']:20s} {dim_tag:5s} {r['val_loss']:8.4f} "
            f"{r['val_exact']:6.3f} {r['meas_loss']:9.4f} {r['meas_exact']:7.3f}  "
            f"m[{mrec}] v[{vrec}]"
        )
    report = "\n".join(lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print("\n" + report)
    print(f"\nWrote report to {REPORT_PATH}")
    print(f"Wrote low-dim models to {OUTPUT_DIR}/ and measure predictions to {PREDICTIONS_DIR}/")


if __name__ == "__main__":
    run()
