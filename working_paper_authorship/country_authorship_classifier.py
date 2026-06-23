"""Censored / chunked working-paper authorship benchmark.

Fetches working papers and their authors, splits deterministically at the document
level, builds censored + uncensored copies at several sentence-chunk granularities,
embeds every chunk, then trains the classifier suite. Hyperparameters are searched once
on the censored full-document set and reused (fixed) for every other dataset. Models,
chosen hyperparameters and a validation report are written to
data/author_classification_models/.
"""
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import optuna
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna_integration import OptunaSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier

from utils import split_parties
from country_meta_info import CaseInsensitiveDict, country_alternative_names
from sentence_splitter import chunk_sentences
from embeddings.working_paper_censorship import get_working_paper_paths, censor_text, llm_censor_text, author_for_stem
from embeddings.document_embeddings import get_wp_ip_embedding_args, get_embedding
from embeddings.embed_all_documents import embed_document_set

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
VAL_TEST_SPLIT_RANDOM_STATE = 7
OPTUNA_RANDOM_STATE = 1234
CV_FOLDS = 3
N_OPTUNA_TRIALS = 100

COUNTRIES = ["Australia", "United Kingdom", "United States", "Norway", "Chile"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]

# Document granularities: "full" document, or fixed-size sentence chunks.
CHUNK_SIZES = [1, 2, 4, 8, 16, 32, 64]
GRANULARITIES = ["full"] + CHUNK_SIZES
# SVM is O(n^2)-ish; only run it where the row count stays modest (big chunks / full doc).
SVM_MIN_CHUNK = 16

# Censorship variants applied to each document's text before chunking/embedding.
CENSORSHIP_METHODS = ["raw", "naive", "llm_censorship"]
# Hyperparameters are searched once on the full documents of this censorship method.
SEARCH_METHOD = "naive"

DOCUMENT_SUMMARY = "data/antarctic-db/processed/document-summary.parquet"
OUTPUT_DIR = pathlib.Path("data/author_classification_models")
N_FEATURES = 4096

_alias_to_canonical = CaseInsensitiveDict()
for _country in COUNTRIES:
    _alias_to_canonical[_country] = _country
    for _alt in country_alternative_names.get(_country, []):
        _alias_to_canonical[_alt] = _country


def parties_to_target_countries(parties) -> set[str]:
    return {
        _alias_to_canonical[p]
        for p in split_parties(parties)
        if p in _alias_to_canonical
    }


# --------------------------------------------------------------------------- metrics

def positive_proba(estimator, X) -> np.ndarray:
    """Return P(label==1) as an (n_samples, n_labels) array, normalising over the
    differing predict_proba conventions: native multilabel trees (XGBoost) return a
    2D array directly, while MultiOutputClassifier / RandomForest return a list of
    (n_samples, n_classes) arrays — one per label."""
    proba = estimator.predict_proba(X)
    if not isinstance(proba, list):
        return np.asarray(proba)

    classes = getattr(estimator, "classes_", None)
    cols = []
    for i, p in enumerate(proba):
        if p.shape[1] >= 2:
            cols.append(p[:, 1])
        else:
            # A CV fold may contain only one class for a rare label (e.g. Norway).
            only_positive = classes is not None and classes[i][0] == 1
            cols.append(np.full(p.shape[0], 1.0 if only_positive else 0.0))
    return np.column_stack(cols)


def mean_cross_entropy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean per-label binary cross-entropy across all country labels."""
    return float(np.mean(np.array([
        log_loss(y_true[:, i], y_proba[:, i], labels=[0, 1])
        for i in range(y_true.shape[1])
    ])))


def neg_cross_entropy_scorer(estimator, X, y) -> float:
    """CV scorer (higher is better) used to select hyperparameters by cross-entropy."""
    return -mean_cross_entropy(y, positive_proba(estimator, X))


# --------------------------------------------------------------------------- models

def _pipe(clf) -> Pipeline:
    return Pipeline([("pca", PCA(random_state=RANDOM_STATE)), ("clf", clf)])


def base_pipeline(name: str) -> Pipeline:
    """A fresh PCA -> classifier pipeline for the named model (untuned)."""
    if name == "Logistic Regression":
        return _pipe(MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)))
    if name == "Random Forest":
        return _pipe(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
    if name == "XGBoost":
        return _pipe(XGBClassifier(
            multi_strategy="multi_output_tree", tree_method="hist",
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
        ))
    if name == "SVM":
        return _pipe(MultiOutputClassifier(CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False)))
    raise ValueError(f"Unknown model: {name}")


def make_search(name: str, pca_dims: list[int]):
    """Wrap the base pipeline in its hyperparameter search (Grid for LR/RF, Optuna for
    XGB/SVM), scored by cross-entropy."""
    pipe = base_pipeline(name)
    if name == "Logistic Regression":
        return GridSearchCV(pipe, {"pca__n_components": pca_dims},
                            scoring=neg_cross_entropy_scorer, cv=CV_FOLDS, n_jobs=-1)
    if name == "Random Forest":
        return GridSearchCV(pipe, {"pca__n_components": pca_dims, "clf__max_depth": [None, 5, 10, 20]},
                            scoring=neg_cross_entropy_scorer, cv=CV_FOLDS, n_jobs=-1)
    if name == "XGBoost":
        return OptunaSearchCV(pipe, {
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__learning_rate": FloatDistribution(1e-2, 3e-1, log=True),
            "clf__max_depth": IntDistribution(3, 10),
            "clf__n_estimators": IntDistribution(100, 1000),
            "clf__subsample": FloatDistribution(0.5, 1.0),
        }, n_trials=N_OPTUNA_TRIALS, scoring=neg_cross_entropy_scorer, cv=CV_FOLDS,
            n_jobs=-1, random_state=OPTUNA_RANDOM_STATE)
    if name == "SVM":
        return OptunaSearchCV(pipe, {
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__estimator__estimator__C": FloatDistribution(1e-2, 1e2, log=True),
            "clf__estimator__estimator__gamma": FloatDistribution(1e-4, 1e0, log=True),
            "clf__estimator__estimator__kernel": CategoricalDistribution(["rbf"]),
        }, n_trials=N_OPTUNA_TRIALS, scoring=neg_cross_entropy_scorer, cv=CV_FOLDS,
            n_jobs=-1, random_state=OPTUNA_RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


def make_fixed(name: str, best_params: dict, n_samples: int) -> Pipeline:
    """Base pipeline with the persisted best params applied, clamping PCA components to
    what this (possibly smaller) dataset can support."""
    pipe = base_pipeline(name)
    params = dict(best_params)
    if "pca__n_components" in params:
        params["pca__n_components"] = max(1, min(params["pca__n_components"], N_FEATURES, n_samples))
    pipe.set_params(**params)
    return pipe


# ----------------------------------------------------------------------------- data

def _build_parties_lookup() -> dict[str, object]:
    """Map a working paper's filename stem -> its `parties` list, from the ATCM WP rows
    of the document-summary parquet (keyed on the paper_url basename stem)."""
    df = pd.read_parquet(DOCUMENT_SUMMARY)
    df = df[(df["meeting_type"] == "ATCM") & (df["party_type"] == "wp")]
    lookup: dict[str, object] = {}
    for row in df.itertuples():
        if isinstance(row.paper_url, str):
            lookup.setdefault(pathlib.Path(row.paper_url).stem, row.parties)
    return lookup


def load_working_papers() -> list[dict]:
    """English working papers authored by >=1 target country, as {stem, text, label, author}.

    ``author`` is the paper's full party string (used to give the LLM censor the known
    authoring party), sourced from the same lookup the censor uses so cache keys line up."""
    lookup = _build_parties_lookup()
    records = []
    for path in get_working_paper_paths():
        parties = lookup.get(path.stem)
        if parties is None:
            # Filenames may carry a revision suffix the parquet stem omits (or vice versa).
            parties = next((p for s, p in lookup.items() if s in path.stem or path.stem in s), None)
        if parties is None or isinstance(parties, float):
            continue
        matched = parties_to_target_countries(parties)
        if not matched:
            continue
        records.append({
            "stem": path.stem,
            "text": path.read_text(encoding="utf-8", errors="ignore"),
            "label": np.array([1 if c in matched else 0 for c in COUNTRIES], dtype=np.int32),
            "author": author_for_stem(path.stem) or ", ".join(str(p) for p in parties),
        })
    return records


def split_documents(records: list[dict]):
    """70 / 15 / 15 train / val / test split at the document level (distinct seeds)."""
    train, temp = train_test_split(records, test_size=0.30, random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.50, random_state=VAL_TEST_SPLIT_RANDOM_STATE)
    return train, val, test


def granularity_label(granularity) -> str:
    return "full" if granularity == "full" else f"chunk{granularity}"


def _chunk_units(text: str, granularity, type_str: str) -> list[tuple]:
    """(hash, type, chunk_text) units for one document at the given granularity. Full
    docs use the whole text; sentence chunks group `granularity` sentences together.
    Either way each chunk is passed through get_wp_ip_embedding_args, which re-splits
    anything over the embedder's context window — a safety net in case the sentence
    tokenizer ever emits a >32k-token "sentence"."""
    chunks = [text] if granularity == "full" else chunk_sentences(text, granularity)
    units = []
    for chunk in chunks:
        if chunk:
            units.extend((h, type_str, seg) for (h, _t, seg) in get_wp_ip_embedding_args(chunk, type_str))
    return units


def _apply_censorship(record: dict, method: str) -> str:
    text = record["text"]
    if method == "raw":
        return text
    if method == "naive":
        return censor_text(text)
    if method == "llm_censorship":
        return llm_censor_text(text, record["author"])
    raise ValueError(f"Unknown censorship method: {method}")


def dataset_units(records: list[dict], method: str, granularity):
    """Return (embed_units, hash_labels) for one (censorship method, granularity) dataset.

    embed_units: list of (hash, type, text) to feed the embedder.
    hash_labels: list of (hash, label) preserving the chunk -> document-label link."""
    type_str = f"WPAuthorClf::{method}::{granularity_label(granularity)}"
    embed_units, hash_labels = [], []
    for rec in records:
        text = _apply_censorship(rec, method)
        for h, t, chunk in _chunk_units(text, granularity, type_str):
            embed_units.append((h, t, chunk))
            hash_labels.append((h, rec["label"]))
    return embed_units, hash_labels


def assemble_xy(hash_labels: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Read cached embeddings back and stack into (X, Y)."""
    X_rows, Y_rows = [], []
    for h, label in hash_labels:
        embedding = get_embedding(h)
        if embedding is None:
            continue
        X_rows.append(embedding)
        Y_rows.append(label)
    return np.array(X_rows, dtype=np.float32), np.array(Y_rows, dtype=np.int32)


# -------------------------------------------------------------------- orchestration

def random_guess_baseline(val_records: list[dict]) -> tuple[float, list[float]]:
    """No-skill BCE baseline: a predictor that outputs each class's validation base rate.
    Returns (mean over classes, per-class). Per-class BCE equals that class's label
    entropy, so the mean is the cross-entropy a prior-only random guess would achieve."""
    labels = np.array([r["label"] for r in val_records])
    base_rates = labels.mean(axis=0)
    proba = np.tile(base_rates, (len(labels), 1))
    per_class = [float(log_loss(labels[:, i], proba[:, i], labels=[0, 1])) for i in range(len(COUNTRIES))]
    return float(np.mean(per_class)), per_class


def _svm_allowed(granularity) -> bool:
    return granularity == "full" or (isinstance(granularity, int) and granularity >= SVM_MIN_CHUNK)


def _model_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _sanitise(obj):
    return obj.item() if hasattr(obj, "item") else obj


def _evaluate(model, X_val, Y_val) -> dict:
    Y_pred = model.predict(X_val)
    Y_proba = positive_proba(model, X_val)
    return {
        "per_country_recall": [float(recall_score(Y_val[:, i], Y_pred[:, i], zero_division=0)) for i in range(len(COUNTRIES))],
        "per_country_precision": [float(precision_score(Y_val[:, i], Y_pred[:, i], zero_division=0)) for i in range(len(COUNTRIES))],
        "exact": float(accuracy_score(Y_val, Y_pred)),
        "loss": mean_cross_entropy(Y_val, Y_proba),
    }


def run_benchmark() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading working papers + authors...")
    records = load_working_papers()
    train, val, test = split_documents(records)
    print(f"  docs: {len(records)} (train {len(train)}, val {len(val)}, test {len(test)} [reserved])")

    datasets = [(method, gran) for method in CENSORSHIP_METHODS for gran in GRANULARITIES]

    # Collect every chunk across all datasets (train + val + test), dedupe, embed once.
    print("Chunking + collecting embedding work...")
    unique_units: dict[str, tuple] = {}
    plans: dict[tuple, list] = {}
    for method, gran in datasets:
        for split_name, recs in (("train", train), ("val", val), ("test", test)):
            embed_units, hash_labels = dataset_units(recs, method, gran)
            for unit in embed_units:
                unique_units.setdefault(unit[0], unit)
            plans[(method, gran, split_name)] = hash_labels
    print(f"Embedding {len(unique_units)} unique chunks (cached ones are skipped)...")
    embed_document_set(list(unique_units.values()))

    # Step 5: search hyperparameters once on the SEARCH_METHOD full-document set. On rerun,
    # reuse the persisted search instead of repeating it.
    hp_path = OUTPUT_DIR / "best_hyperparameters.json"
    if hp_path.exists():
        print(f"\nLoading cached hyperparameters from {hp_path} (skipping search)...")
        best_params = json.loads(hp_path.read_text())
    else:
        Xc_train, Yc_train = assemble_xy(plans[(SEARCH_METHOD, "full", "train")])
        pca_dims = pca_search_dims(len(Xc_train), N_FEATURES)
        print(f"\nSearching hyperparameters on {SEARCH_METHOD} full docs "
              f"(n_train={len(Xc_train)}, pca_dims={pca_dims})...")
        best_params = {}
        for name in MODEL_NAMES:
            print(f"  searching {name}...")
            search = make_search(name, pca_dims)
            search.fit(Xc_train, Yc_train)
            best_params[name] = {k: _sanitise(v) for k, v in search.best_params_.items()}
        hp_path.write_text(json.dumps(best_params, indent=2))

    # Step 6: for every (dataset, model), reuse the saved model if it exists (rerun),
    # otherwise fit it with the fixed hyperparameters and persist it. Validation
    # statistics are recomputed either way.
    results = []
    for method, gran in datasets:
        tag = f"{method}/{granularity_label(gran)}"
        X_val, Y_val = assemble_xy(plans[(method, gran, "val")])
        X_train = Y_train = None  # assembled lazily, only when a model needs fitting
        print(f"\nDataset {tag}: val {X_val.shape}")
        for name in MODEL_NAMES:
            if name == "SVM" and not _svm_allowed(gran):
                continue
            slug = f"{_model_slug(name)}__{method}__{granularity_label(gran)}"
            pickle_path = OUTPUT_DIR / f"{slug}.pickle"
            if pickle_path.exists():
                with open(pickle_path, "rb") as f:
                    model = pickle.load(f)
            else:
                if X_train is None:
                    X_train, Y_train = assemble_xy(plans[(method, gran, "train")])
                model = make_fixed(name, best_params[name], X_train.shape[0])
                model.fit(X_train, Y_train)
                with open(pickle_path, "wb") as f:
                    pickle.dump(model, f)

            metrics = _evaluate(model, X_val, Y_val)
            results.append({"model": name, "method": method, "granularity": gran, **metrics})
            print(f"  {name:20s} loss={metrics['loss']:.4f} exact={metrics['exact']:.4f}")

    baseline_avg, baseline_per_class = random_guess_baseline(val)
    write_report(results, baseline_avg, baseline_per_class)
    return results


def pca_search_dims(n_train: int, n_features: int) -> list[int]:
    """Powers of two up to 4096, capped at what a CV training fold can support
    (n_components <= min(n_features, fold samples))."""
    max_components = min(n_features, (n_train * (CV_FOLDS - 1)) // CV_FOLDS)
    return [d for d in (2 ** i for i in range(13)) if d <= max_components]


def write_report(results: list[dict], baseline_avg: float, baseline_per_class: list[float]) -> None:
    baseline_cols = " ".join(f"{b:.2f}" for b in baseline_per_class)
    lines = ["WORKING PAPER AUTHORSHIP — CENSORED / CHUNKED BENCHMARK",
             f"Countries: {', '.join(COUNTRIES)}",
             "Metrics on the validation set (cross-entropy lower is better).",
             f"Random-guess BCE baseline (predict each class's base rate): {baseline_avg:.4f}  per-class[{baseline_cols}]",
             f"Per-country recall / precision order: {', '.join(COUNTRIES)}",
             "",
             f"{'model':20s} {'method':15s} {'gran':7s} {'x-entropy':>10s} {'exact':>7s}  per-country recall / precision"]
    for r in sorted(results, key=lambda r: (r["model"], CENSORSHIP_METHODS.index(r["method"]), str(r["granularity"]))):
        rec = " ".join(f"{x:.2f}" for x in r["per_country_recall"])
        prec = " ".join(f"{p:.2f}" for p in r["per_country_precision"])
        lines.append(f"{r['model']:20s} {r['method']:15s} "
                     f"{granularity_label(r['granularity']):7s} {r['loss']:10.4f} {r['exact']:7.4f}  "
                     f"rec[{rec}] prec[{prec}]")
    report = "\n".join(lines)
    (OUTPUT_DIR / "report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote report + {len(results)} models + best_hyperparameters.json to {OUTPUT_DIR}/")


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
