"""Working-paper authorship benchmark over whole-document representations.

Fetches working papers and their authors, splits deterministically at the document level,
then builds and embeds one whole-document representation per censorship method (raw / naive /
LLM). The classifier suite is trained on each. Hyperparameters are searched *separately* for every
dataset (no shared search). Models, per-dataset hyperparameters and a validation report are
written to data/author_classification_models/, and a per-(model, dataset) CSV of held-out test
set predictions to data/test_set_predictions/.

Pass --final-test-eval to add a last stage: each dataset's best-on-validation model is refit on
train+val (same hyperparameters, no new search) and scored once on the held-out test set, into
final_test_report.txt. It is opt-in because every run of it consumes the test set.
"""
import argparse
import hashlib
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
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier

from utils import split_parties, line_buffer_stdout
from country_meta_info import CaseInsensitiveDict, country_alternative_names
from embeddings.working_paper_censorship import get_working_paper_paths, censor_text, llm_censor_text, author_for_stem
from embeddings.document_embeddings import (add_document_type, get_wp_ip_embedding_args,
                                            get_embedding, has_embedding)
from embeddings.embed_all_documents import embed_document_set
from working_paper_authorship.country_signal_projection import CountrySignalProjector
from working_paper_authorship.authorship_performance_figures import render_all_figures

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
VAL_TEST_SPLIT_RANDOM_STATE = 7
OPTUNA_RANDOM_STATE = 1234
CV_FOLDS = 3
N_OPTUNA_TRIALS = 32
# Parallel candidates/trials per search. Set to 1 on purpose: one candidate already saturates the
# machine — its PCA SVD streams the whole n x 4096 document matrix (memory-bandwidth bound)
# and, with the estimators at n_jobs=-1, uses all cores. Fitting candidates one at a time (each
# using the whole machine) avoids the cross-candidate memory-bandwidth contention, peak-RAM spikes,
# and joblib memmap/IPC overhead that n_jobs=-1 caused here. Estimators parallelise internally.
SEARCH_N_JOBS = 1

COUNTRIES = ["Australia", "United Kingdom", "United States", "Norway", "Chile"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]

# The datasets we benchmark, each (slug, censorship method): whole documents under every
# censorship method. Hyperparameters are searched separately for each dataset.
DATASETS = [
    ("raw__full",            "raw"),
    ("naive__full",          "naive"),
    ("llm_censorship__full", "llm_censorship"),
]

DOCUMENT_SUMMARY = "data/antarctic-db/processed/document-summary.parquet"
OUTPUT_DIR = pathlib.Path("data/author_classification_models")
PREDICTIONS_DIR = pathlib.Path("data/test_set_predictions")
# --orthogonalize-country: train on embeddings with the direct country-signal subspace projected out.
# Separate output dirs so the orthogonal experiment never overwrites the full-space baseline models.
COUNTRY_DIRECTIONS_PATH = pathlib.Path("data/country_signal/direct_country_directions_allwps.npz")
ORTHOGONAL_OUTPUT_DIR = pathlib.Path("data/author_classification_models_orthogonal")
ORTHOGONAL_PREDICTIONS_DIR = pathlib.Path("data/test_set_predictions_orthogonal")
_PROJECTOR = None  # set by run_benchmark when orthogonalizing; assemble_xy applies it to every X
N_FEATURES = 4096
# Cap on PCA components searched. >512 components rarely helps, and capping here keeps the search
# space consistent across datasets (the whole-doc sets already topped out at 512 via the fold-size
# limit below).
MAX_PCA_COMPONENTS = 512

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

    # Which class a single-column predict_proba refers to has to come from the per-label estimator,
    # not from the wrapper. MultiOutputClassifier exposes no `classes_` of its own (sklearn 1.9),
    # so the old `getattr(estimator, "classes_", None)` was always None and the branch below always
    # took the negative reading -- a fold containing only positives for a rare label (Norway is the
    # one that hits this) was scored as P(author)=0.0 for every row, i.e. maximally wrong, and the
    # cross-entropy that selected hyperparameters was computed against it.
    estimators = getattr(estimator, "estimators_", None)

    cols = []
    for i, p in enumerate(proba):
        if p.shape[1] >= 2:
            cols.append(p[:, 1])
        else:
            # A CV fold may contain only one class for a rare label (e.g. Norway). The single
            # column is P(that class), so it is P(label==1) exactly when that class *is* 1.
            classes = getattr(estimators[i], "classes_", None) if estimators is not None else None
            only_positive = classes is not None and len(classes) and classes[0] == 1
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
        # n_jobs=-1: the search runs one candidate at a time (SEARCH_N_JOBS=1), so each RF should
        # use all cores itself rather than run single-threaded.
        return _pipe(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
    if name == "XGBoost":
        return _pipe(XGBClassifier(
            multi_strategy="multi_output_tree", tree_method="hist",
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ))
    if name == "SVM":
        return _pipe(MultiOutputClassifier(CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False)))
    raise ValueError(f"Unknown model: {name}")


def inner_cv():
    """The cross-validation splitter used inside the hyperparameter search.

    Grouped by source document, not by row. A paper longer than the embedder's context window is
    split by ``get_wp_ip_embedding_args`` into several segments, and every segment carries that
    paper's authorship label -- so a plain row-wise ``KFold`` can put one part of a paper in train
    and another in validation. The two are near-duplicate text with an identical label, which is
    the textbook leak: the score it produces is optimistic, and it is the score the hyperparameters
    are selected on.

    Today's rows happen to be contiguous per document, so an unshuffled KFold only leaked at fold
    boundaries and the damage was small -- but nothing enforced that ordering, and grouping states
    the requirement instead of relying on it. ``stems`` out of ``assemble_xy`` is the group key.

    GroupKFold is also deterministic (it has no shuffle), which keeps the search reproducible.
    """
    return GroupKFold(n_splits=CV_FOLDS)


def make_search(name: str, pca_dims: list[int]):
    """Wrap the base pipeline in its hyperparameter search (Grid for LR/RF, Optuna for
    XGB/SVM), scored by cross-entropy.

    The returned search must be fitted with ``groups=`` (the per-row document stems) -- see
    ``inner_cv``."""
    pipe = base_pipeline(name)
    if name == "Logistic Regression":
        return GridSearchCV(pipe, {"pca__n_components": pca_dims},
                            scoring=neg_cross_entropy_scorer, cv=inner_cv(), n_jobs=SEARCH_N_JOBS)
    if name == "Random Forest":
        return GridSearchCV(pipe, {"pca__n_components": pca_dims, "clf__max_depth": [None, 5, 10, 20]},
                            scoring=neg_cross_entropy_scorer, cv=inner_cv(), n_jobs=SEARCH_N_JOBS)
    if name == "XGBoost":
        return OptunaSearchCV(pipe, {
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__learning_rate": FloatDistribution(1e-2, 3e-1, log=True),
            "clf__max_depth": IntDistribution(3, 10),
            "clf__n_estimators": IntDistribution(100, 1000),
            "clf__subsample": FloatDistribution(0.5, 1.0),
        }, n_trials=N_OPTUNA_TRIALS, scoring=neg_cross_entropy_scorer, cv=inner_cv(),
            n_jobs=SEARCH_N_JOBS, random_state=OPTUNA_RANDOM_STATE)
    if name == "SVM":
        return OptunaSearchCV(pipe, {
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__estimator__estimator__C": FloatDistribution(1e-2, 1e2, log=True),
            "clf__estimator__estimator__gamma": FloatDistribution(1e-4, 1e0, log=True),
            "clf__estimator__estimator__kernel": CategoricalDistribution(["rbf"]),
        }, n_trials=N_OPTUNA_TRIALS, scoring=neg_cross_entropy_scorer, cv=inner_cv(),
            n_jobs=SEARCH_N_JOBS, random_state=OPTUNA_RANDOM_STATE)
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


def _apply_censorship(record: dict, method: str) -> str:
    text = record["text"]
    if method == "raw":
        return text
    if method == "naive":
        return censor_text(text)
    if method == "llm_censorship":
        return llm_censor_text(text, record["author"])
    raise ValueError(f"Unknown censorship method: {method}")


def dataset_units(records: list[dict], method: str):
    """Return (embed_units, hash_labels) for one censorship method's whole-document dataset.

    embed_units: list of (hash, type, text) to feed the embedder.
    hash_labels: list of (hash, label, stem) preserving the fragment -> document link (its label
    and which working paper it came from).

    One unit per document, except that get_wp_ip_embedding_args re-splits anything over the
    embedder's context window, so a handful of very long papers contribute more than one row."""
    type_str = f"WPAuthorClf::{method}::full"
    embed_units, hash_labels = [], []
    for rec in records:
        text = _apply_censorship(rec, method)
        for h, t, segment in get_wp_ip_embedding_args(text, type_str):
            embed_units.append((h, t, segment))
            hash_labels.append((h, rec["label"], rec["stem"]))
    return embed_units, hash_labels


def dataset_fingerprint(*hash_label_plans) -> str:
    """A content fingerprint of one dataset: its rows, their labels and which document each is from.

    Every cached artefact here -- ``best_hyperparameters__{slug}.json`` and the model pickles -- is
    keyed on the dataset *slug* alone, which names the censorship method and nothing about the data.
    So a rerun after the corpus changed (a re-OCR, a re-embed, a censorship fix, a different
    train/val split) loaded hyperparameters searched on the old data and models fitted to it, and
    reported their numbers as if they described the new data. Nothing anywhere said otherwise.

    The fingerprint closes that. It hashes the row hashes -- which are sha256 of the *censored*
    text, so they move if either the source text or the censorship changes -- together with each
    row's label and stem, so a relabelling or a reshuffled split is caught too. Sorted first, so it
    depends on the dataset's content rather than on enumeration order.
    """
    digest = hashlib.sha256()
    rows = sorted(
        (h, stem, ",".join(str(int(v)) for v in np.asarray(label).ravel()))
        for plan in hash_label_plans for h, label, stem in plan
    )
    for h, stem, label in rows:
        digest.update(f"{h}\x1f{stem}\x1f{label}\x1e".encode())
    return digest.hexdigest()


def cached_artefacts_are_current(slug: str, fingerprint: str, output_dir: pathlib.Path) -> bool:
    """Whether ``slug``'s cached models/hyperparameters describe this exact dataset.

    True when no fingerprint has been recorded yet (a first run, or a cache from before this
    existed) -- unknown provenance is not proof of staleness, and refusing to use a cache we have
    no evidence against would throw away every existing model for nothing. The fingerprint is
    written after a successful run, so the check has teeth from then on.
    """
    path = output_dir / f"dataset_fingerprint__{slug}.json"
    if not path.exists():
        return True
    return json.loads(path.read_text()).get("fingerprint") == fingerprint


def write_dataset_fingerprint(slug: str, fingerprint: str, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"dataset_fingerprint__{slug}.json").write_text(
        json.dumps({"fingerprint": fingerprint}, indent=2))


def collect_unique_units(all_units) -> tuple[dict, dict]:
    """Dedupe embedding work by hash, remembering every type each hash appeared under.

    A unit is ``(hash, type_str, text)`` and the hash is sha256 of the *censored* text, while the
    type is ``WPAuthorClf::{method}::full`` -- so for a paper that names none of the target
    countries, censorship is a no-op and all three datasets produce one hash under three different
    types. Deduping is right (embedding it three times would be three identical paid calls for one
    vector), but a plain ``setdefault`` also discards the two type strings that lost, and since
    DATASETS is ordered raw-first, raw always won.

    That mattered because the discarded types were never *submitted*: ``get_or_generate_embedding``
    records a type on a cache hit precisely so types accumulate, but it only ever saw one unit per
    hash, so the accumulation had nothing to accumulate. The result was the same defect the
    document_type array was introduced to fix, re-introduced one level above the storage layer --
    ``get_embeddings_by_type("WPAuthorClf::naive::full")`` silently omitted every document
    censorship left unchanged.

    Training was never affected: ``assemble_xy`` fetches by hash and ignores types entirely. What
    was affected is everything that *enumerates* by type, such as ``export_document_embeddings``.

    Returns ``(unique_units, seen_types)``; pass both to ``register_dropped_types`` after embedding.
    """
    unique_units: dict[str, tuple] = {}
    seen_types: dict[str, set[str]] = {}
    for unit in all_units:
        digest, type_str, _text = unit
        unique_units.setdefault(digest, unit)
        seen_types.setdefault(digest, set()).add(type_str)
    return unique_units, seen_types


def register_dropped_types(unique_units: dict, seen_types: dict) -> int:
    """Record the dataset types that deduping discarded.

    Must run *after* the embedding pass: ``add_document_type`` tags an existing row and does
    nothing when there is none.

    Returns the number of (hash, type) pairs tagged -- pairs, not *newly added* types.
    ``add_document_type`` reports whether the row existed, not whether the type was new to it, and
    the tagging is idempotent, so a rerun tags the same pairs again and returns the same count.
    Distinguishing the two would cost a query per pair for a number that only feeds a log line.
    """
    added = 0
    for digest, types in seen_types.items():
        embedded_as = unique_units[digest][1]
        for type_str in sorted(types - {embedded_as}):
            if add_document_type(digest, type_str):
                added += 1
    return added


def assemble_xy(hash_labels: list[tuple]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read cached embeddings back and stack into (X, Y, stems), where ``stems`` names the source
    working paper of each row. Raises if any expected embedding is missing rather than silently
    dropping the row — a partial embedding cache must never quietly shrink/skew a dataset. Run the
    embedding step (or embed_missing_embeddings) first."""
    X_rows, Y_rows, stems, missing = [], [], [], 0
    for h, label, stem in hash_labels:
        embedding = get_embedding(h)
        if embedding is None:
            missing += 1
            continue
        X_rows.append(embedding)
        Y_rows.append(label)
        stems.append(stem)
    if missing:
        raise RuntimeError(
            f"{missing}/{len(hash_labels)} fragments have no cached embedding. Re-run the "
            f"embedding step (run_benchmark embeds automatically; or call embed_missing_embeddings)."
        )
    X = np.array(X_rows, dtype=np.float32)
    if _PROJECTOR is not None:
        # Project every embedding onto the complement of the direct country-signal subspace.
        X = _PROJECTOR.transform(X)
    return X, np.array(Y_rows, dtype=np.int32), stems


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


def model_slug(name: str) -> str:
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


def write_test_predictions(model, name: str, slug: str, X_test, Y_test, stems: list[str]) -> pathlib.Path:
    """Write one CSV per (model, dataset) holding every test-set working paper: its true labels,
    the model's predicted labels, and the pseudo-probabilities behind them.

    One row per working paper. For the per-sentence dataset a document contributes many rows to
    X_test, so its probabilities are averaged over its sentences and the prediction re-derived by
    thresholding that mean at 0.5 (``n_units`` records how many sentences were pooled); for the
    whole-document datasets n_units is 1 and the probabilities/predictions are the model's own."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    proba = positive_proba(model, X_test)

    df = pd.DataFrame({"stem": stems})
    for i, country in enumerate(COUNTRIES):
        df[f"true__{country}"] = Y_test[:, i]
        df[f"prob__{country}"] = proba[:, i]
    df["n_units"] = 1

    prob_cols = [f"prob__{c}" for c in COUNTRIES]
    # A document's true labels are identical across its rows, so "first" and "mean" agree there.
    agg = {**{f"true__{c}": "first" for c in COUNTRIES}, **{c: "mean" for c in prob_cols}, "n_units": "sum"}
    df = df.groupby("stem", as_index=False, sort=True).agg(agg)
    for country in COUNTRIES:
        df[f"pred__{country}"] = (df[f"prob__{country}"] >= 0.5).astype(int)

    columns = ["stem", "n_units"] + [f"{p}__{c}" for c in COUNTRIES for p in ("true", "pred", "prob")]
    path = PREDICTIONS_DIR / f"{model_slug(name)}__{slug}.csv"
    df[columns].to_csv(path, index=False)
    return path


def run_final_test_evaluation(results: list[dict], plans: dict, val_records: list[dict]) -> list[dict]:
    """Refit each dataset's best-on-validation model on train + val, then score it on the test set.

    This is the one place the held-out test set is scored, and it is deliberately last: the winner
    per dataset is chosen on validation cross-entropy only, and no hyperparameter is re-searched
    here — the dataset's cached best params are reused, refit on the larger train+val sample.
    Final models are persisted under a ``final__`` prefix so they never overwrite the train-only
    models the validation benchmark reports on."""
    by_dataset: dict[str, dict] = {}
    for r in results:
        if r["dataset"] not in by_dataset or r["loss"] < by_dataset[r["dataset"]]["loss"]:
            by_dataset[r["dataset"]] = r

    final_results = []
    for slug, _method in DATASETS:
        winner = by_dataset.get(slug)
        if winner is None:
            continue
        name = winner["model"]
        best_params = json.loads((OUTPUT_DIR / f"best_hyperparameters__{slug}.json").read_text())

        X_trainval, Y_trainval, _ = assemble_xy(plans[(slug, "train")] + plans[(slug, "val")])
        X_test, Y_test, test_stems = assemble_xy(plans[(slug, "test")])

        pickle_path = OUTPUT_DIR / f"final__{model_slug(name)}__{slug}.pickle"
        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                model = pickle.load(f)
        else:
            print(f"  refitting {name} on train+val for {slug} ({X_trainval.shape[0]} rows)...")
            model = make_fixed(name, best_params[name], X_trainval.shape[0])
            model.fit(X_trainval, Y_trainval)
            with open(pickle_path, "wb") as f:
                pickle.dump(model, f)

        metrics = _evaluate(model, X_test, Y_test)
        write_test_predictions(model, f"final__{name}", slug, X_test, Y_test, test_stems)
        final_results.append({"model": name, "dataset": slug, "val_loss": winner["loss"], **metrics})
        print(f"  {slug:22s} best={name:20s} val_loss={winner['loss']:.4f} -> test_loss={metrics['loss']:.4f}")

    # Baseline base rates come from val, not test — the no-skill reference should not be tuned to
    # the very labels it is a reference for.
    baseline_avg, baseline_per_class = random_guess_baseline(val_records)
    write_final_test_report(final_results, baseline_avg, baseline_per_class)
    return final_results


def write_final_test_report(results: list[dict], baseline_avg: float, baseline_per_class: list[float]) -> None:
    baseline_cols = " ".join(f"{b:.2f}" for b in baseline_per_class)
    dataset_order = {slug: i for i, (slug, _) in enumerate(DATASETS)}
    lines = ["WORKING PAPER AUTHORSHIP — FINAL HELD-OUT TEST EVALUATION",
             f"Countries: {', '.join(COUNTRIES)}",
             "Per dataset: the model with the best validation cross-entropy, refit on train+val",
             "and scored once on the held-out test set. Hyperparameters were not re-searched.",
             f"Random-guess BCE baseline (validation base rates): {baseline_avg:.4f}  per-class[{baseline_cols}]",
             f"Per-country recall / precision order: {', '.join(COUNTRIES)}",
             "",
             f"{'dataset':22s} {'best model':20s} {'val x-ent':>9s} {'test x-ent':>10s} {'test exact':>10s}  per-country recall / precision"]
    ordered = sorted(results, key=lambda r: dataset_order[r["dataset"]])
    for r in ordered:
        rec = " ".join(f"{x:.2f}" for x in r["per_country_recall"])
        prec = " ".join(f"{p:.2f}" for p in r["per_country_precision"])
        lines.append(f"{r['dataset']:22s} {r['model']:20s} {r['val_loss']:9.4f} "
                     f"{r['loss']:10.4f} {r['exact']:10.4f}  rec[{rec}] prec[{prec}]")
    if ordered:
        overall = min(ordered, key=lambda r: r["val_loss"])
        lines += ["",
                  f"Best overall on validation: {overall['model']} on {overall['dataset']} "
                  f"(val {overall['val_loss']:.4f}) -> test cross-entropy {overall['loss']:.4f}"]
    report = "\n".join(lines)
    (OUTPUT_DIR / "final_test_report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote final test report to {OUTPUT_DIR}/final_test_report.txt")


def run_benchmark(final_test_eval: bool = False, orthogonalize_country: bool = False) -> list[dict]:
    global _PROJECTOR, OUTPUT_DIR, PREDICTIONS_DIR
    if orthogonalize_country:
        _PROJECTOR = CountrySignalProjector.from_npz(COUNTRY_DIRECTIONS_PATH)
        print(f"Orthogonalizing embeddings against the direct country signal: projecting out a "
              f"rank-{_PROJECTOR.rank} subspace ({COUNTRY_DIRECTIONS_PATH}).")
        baseline_dir = OUTPUT_DIR
        OUTPUT_DIR = ORTHOGONAL_OUTPUT_DIR
        PREDICTIONS_DIR = ORTHOGONAL_PREDICTIONS_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Reuse the baseline (full-space) hyperparameters instead of re-searching, so the only thing that
        # changes between baseline and this run is the feature space — a clean A/B. Copy them across once.
        for slug, _m in DATASETS:
            src = baseline_dir / f"best_hyperparameters__{slug}.json"
            dst = OUTPUT_DIR / f"best_hyperparameters__{slug}.json"
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading working papers + authors...")
    records = load_working_papers()
    train, val, test = split_documents(records)
    print(f"  docs: {len(records)} (train {len(train)}, val {len(val)}, test {len(test)} [held out])")

    # Collect every chunk across all datasets (train + val + test), dedupe, embed once.
    print("Chunking + collecting embedding work...")
    all_units: list[tuple] = []
    plans: dict[tuple, list] = {}
    for slug, method in DATASETS:
        for split_name, recs in (("train", train), ("val", val), ("test", test)):
            embed_units, hash_labels = dataset_units(recs, method)
            all_units.extend(embed_units)
            plans[(slug, split_name)] = hash_labels
    unique_units, seen_types = collect_unique_units(all_units)
    print(f"Embedding {len(unique_units)} unique chunks (cached ones are skipped)...")
    embed_document_set(list(unique_units.values()))

    # Chunks shared between datasets (censorship was a no-op) were embedded under whichever
    # dataset's type came first; tag them with the others now that the rows exist, or those
    # datasets are unenumerable by their own type. See collect_unique_units.
    added = register_dropped_types(unique_units, seen_types)
    if added:
        print(f"  tagged {added} shared chunk/type pair(s) that deduping would otherwise have lost")

    # Guard: every fragment must be embedded before training, so a partial cache can't silently
    # shrink a dataset (assemble_xy would otherwise drop the missing rows).
    missing = [h for h in unique_units if not has_embedding(h)]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(unique_units)} fragments still have no embedding after "
            f"embed_document_set — aborting rather than training on a partial dataset."
        )

    # For every dataset, search its own hyperparameters (cached per dataset, reused on rerun),
    # then fit + persist each model with those params. Validation statistics are recomputed
    # either way. Saved models are reused as-is on rerun.
    results = []
    for slug, method in DATASETS:
        # train_stems groups the rows by source document for the inner CV -- see inner_cv().
        X_train, Y_train, train_stems = assemble_xy(plans[(slug, "train")])
        X_val, Y_val, _ = assemble_xy(plans[(slug, "val")])
        X_test, Y_test, test_stems = assemble_xy(plans[(slug, "test")])
        print(f"\nDataset {slug}: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

        # Cached hyperparameters and models are keyed on the slug, which says nothing about the
        # data. Check they were produced from this exact dataset before trusting them; if not,
        # ignore them and retrain rather than silently reporting numbers from the old corpus.
        fingerprint = dataset_fingerprint(plans[(slug, "train")], plans[(slug, "val")],
                                          plans[(slug, "test")])
        reusable = cached_artefacts_are_current(slug, fingerprint, OUTPUT_DIR)
        if not reusable:
            print(f"  !! {slug} changed since its cached models were built — re-searching "
                  f"hyperparameters and refitting instead of reusing them.")

        hp_path = OUTPUT_DIR / f"best_hyperparameters__{slug}.json"
        if hp_path.exists() and reusable:
            print(f"  loading cached hyperparameters from {hp_path.name} (skipping search)")
            best_params = json.loads(hp_path.read_text())
        else:
            pca_dims = pca_search_dims(len(X_train), N_FEATURES)
            print(f"  searching hyperparameters (n_train={len(X_train)}, pca_dims={pca_dims})...")
            best_params = {}
            for name in MODEL_NAMES:
                print(f"    searching {name}...")
                search = make_search(name, pca_dims)
                search.fit(X_train, Y_train, groups=train_stems)
                best_params[name] = {k: _sanitise(v) for k, v in search.best_params_.items()}
            hp_path.write_text(json.dumps(best_params, indent=2))

        for name in MODEL_NAMES:
            pickle_path = OUTPUT_DIR / f"{model_slug(name)}__{slug}.pickle"
            if pickle_path.exists() and reusable:
                with open(pickle_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = make_fixed(name, best_params[name], X_train.shape[0])
                model.fit(X_train, Y_train)
                with open(pickle_path, "wb") as f:
                    pickle.dump(model, f)

            metrics = _evaluate(model, X_val, Y_val)
            results.append({"model": name, "dataset": slug, **metrics})
            pred_path = write_test_predictions(model, name, slug, X_test, Y_test, test_stems)
            print(f"  {name:20s} loss={metrics['loss']:.4f} exact={metrics['exact']:.4f}"
                  f"  test predictions -> {pred_path.name}")

        # Written only once every model for this dataset is fitted and persisted, so an interrupted
        # run leaves no fingerprint vouching for a cache that is only half there.
        write_dataset_fingerprint(slug, fingerprint, OUTPUT_DIR)

    baseline_avg, baseline_per_class = random_guess_baseline(val)
    write_report(results, baseline_avg, baseline_per_class)
    # Figures are rendered from the persisted reports of *both* runs (full-space and orthogonal),
    # not from this run's in-memory results, so the comparison figure appears once both exist.
    render_all_figures()

    if final_test_eval:
        print("\nFinal held-out test evaluation (refitting each dataset's best model on train+val)...")
        run_final_test_evaluation(results, plans, val)
    return results


def pca_search_dims(n_train: int, n_features: int) -> list[int]:
    """Powers of two, capped at MAX_PCA_COMPONENTS and at what a CV training fold can support
    (n_components <= min(n_features, fold samples))."""
    max_components = min(n_features, MAX_PCA_COMPONENTS, (n_train * (CV_FOLDS - 1)) // CV_FOLDS)
    return [d for d in (2 ** i for i in range(13)) if d <= max_components]


def write_report(results: list[dict], baseline_avg: float, baseline_per_class: list[float]) -> None:
    baseline_cols = " ".join(f"{b:.2f}" for b in baseline_per_class)
    dataset_order = {slug: i for i, (slug, _) in enumerate(DATASETS)}
    lines = ["WORKING PAPER AUTHORSHIP — WHOLE-DOC vs KEEP-SENTENCE BENCHMARK",
             f"Countries: {', '.join(COUNTRIES)}",
             "Metrics on the validation set (cross-entropy lower is better).",
             f"Random-guess BCE baseline (predict each class's base rate): {baseline_avg:.4f}  per-class[{baseline_cols}]",
             f"Per-country recall / precision order: {', '.join(COUNTRIES)}",
             "",
             f"{'model':20s} {'dataset':22s} {'x-entropy':>10s} {'exact':>7s}  per-country recall / precision"]
    for r in sorted(results, key=lambda r: (r["model"], dataset_order[r["dataset"]])):
        rec = " ".join(f"{x:.2f}" for x in r["per_country_recall"])
        prec = " ".join(f"{p:.2f}" for p in r["per_country_precision"])
        lines.append(f"{r['model']:20s} {r['dataset']:22s} "
                     f"{r['loss']:10.4f} {r['exact']:7.4f}  rec[{rec}] prec[{prec}]")
    report = "\n".join(lines)
    (OUTPUT_DIR / "report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote report + {len(results)} models + per-dataset hyperparameters to {OUTPUT_DIR}/")
    print(f"Wrote {len(results)} test-set prediction CSVs to {PREDICTIONS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--final-test-eval", action="store_true",
        help="After the validation benchmark, refit each dataset's best-on-validation model on "
             "train+val and score it once on the held-out test set, writing final_test_report.txt. "
             "Off by default: every run of this touches the test set.",
    )
    parser.add_argument(
        "--orthogonalize-country", action="store_true",
        help="Train on embeddings with the direct country-signal subspace (from the injection probe) "
             "projected out, to test whether removing the direct signal generalizes better. Writes to "
             "separate *_orthogonal dirs and reuses the baseline hyperparameters (feature space is the "
             "only thing that changes).",
    )
    args = parser.parse_args()
    run_benchmark(final_test_eval=args.final_test_eval, orthogonalize_country=args.orthogonalize_country)


if __name__ == "__main__":
    line_buffer_stdout()
    main()
