import sys
import pathlib
import pickle

import numpy as np
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
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from embeddings.document_embeddings import DocumentTextGetter
from utils import split_parties
from country_meta_info import CaseInsensitiveDict, country_alternative_names
from working_paper_authorship.torch_svm import RFFSVM

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
VAL_TEST_SPLIT_RANDOM_STATE = 7
OPTUNA_RANDOM_STATE = 1234
CV_FOLDS = 3
N_OPTUNA_TRIALS = 100

COUNTRIES = ["Australia", "United Kingdom", "United States", "Norway", "Chile"]

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


def build_models(pca_dims: list[int]) -> list[dict]:
    """Every model is a PCA -> classifier pipeline (steps named "pca" and "clf"),
    hyperparameter-tuned with k-fold CV against cross-entropy. Native multilabel
    classifiers are used where available (RandomForest, XGBoost); logistic regression
    and SVM have no native multilabel support so they are wrapped in
    MultiOutputClassifier."""

    def pipe(clf) -> Pipeline:
        return Pipeline([("pca", PCA(random_state=RANDOM_STATE)), ("clf", clf)])

    logistic = GridSearchCV(
        pipe(MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))),
        param_grid={
            "pca__n_components": pca_dims,
        },
        scoring=neg_cross_entropy_scorer,
        cv=CV_FOLDS,
        n_jobs=-1,
    )

    random_forest = GridSearchCV(
        pipe(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        param_grid={
            "pca__n_components": pca_dims,
            "clf__max_depth": [None, 5, 10, 20],
        },
        scoring=neg_cross_entropy_scorer,
        cv=CV_FOLDS,
        n_jobs=-1,
    )

    # OptunaSearchCV runs Bayesian (TPE) hyperparameter optimisation while keeping the
    # familiar sklearn fit / predict / best_params_ interface.
    xgboost = OptunaSearchCV(
        pipe(XGBClassifier(
            multi_strategy="multi_output_tree",
            tree_method="hist",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
        param_distributions={
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__learning_rate": FloatDistribution(1e-2, 3e-1, log=True),
            "clf__max_depth": IntDistribution(3, 10),
            "clf__n_estimators": IntDistribution(100, 1000),
            "clf__subsample": FloatDistribution(0.5, 1.0),
        },
        n_trials=N_OPTUNA_TRIALS,
        scoring=neg_cross_entropy_scorer,
        cv=CV_FOLDS,
        n_jobs=-1,
        random_state=OPTUNA_RANDOM_STATE,
    )

    # SVC has no native probability output; CalibratedClassifierCV (the sklearn-1.9+
    # replacement for SVC(probability=True)) gives calibrated probabilities for the
    # cross-entropy scorer, and MultiOutputClassifier extends it to multilabel.
    svm = OptunaSearchCV(
        pipe(MultiOutputClassifier(CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False))),
        param_distributions={
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__estimator__estimator__C": FloatDistribution(1e-2, 1e2, log=True),
            "clf__estimator__estimator__gamma": FloatDistribution(1e-4, 1e0, log=True),
            "clf__estimator__estimator__kernel": CategoricalDistribution(["rbf"]),
        },
        n_trials=N_OPTUNA_TRIALS,
        scoring=neg_cross_entropy_scorer,
        cv=CV_FOLDS,
        n_jobs=-1,
        random_state=OPTUNA_RANDOM_STATE,
    )

    # RFFSVM approximates the rbf kernel with random Fourier features and is natively
    # multilabel and differentiable; included here to compare against the rbf SVC above.
    differentiable_svm = OptunaSearchCV(
        pipe(RFFSVM(random_state=RANDOM_STATE)),
        param_distributions={
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__C": FloatDistribution(1e-2, 1e2, log=True),
            "clf__gamma": FloatDistribution(1e-3, 1e0, log=True),
            "clf__n_features": CategoricalDistribution([256, 512, 1024]),
        },
        n_trials=N_OPTUNA_TRIALS,
        scoring=neg_cross_entropy_scorer,
        cv=CV_FOLDS,
        n_jobs=-1,
        random_state=OPTUNA_RANDOM_STATE,
    )

    return [
        {"name": "Logistic Regression", "model": logistic},
        {"name": "Random Forest", "model": random_forest},
        {"name": "XGBoost", "model": xgboost},
        {"name": "SVM", "model": svm},
        {"name": "Differentiable RFF-SVM", "model": differentiable_svm},
    ]


def build_dataset(docs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X_rows, Y_rows = [], []

    for doc in docs:
        parties = doc.get("parties")
        embedding = doc.get("embedding")
        if parties is None or embedding is None:
            continue

        matched = parties_to_target_countries(parties)
        if not matched:
            continue

        X_rows.append(embedding)
        Y_rows.append([1 if c in matched else 0 for c in COUNTRIES])

    return np.array(X_rows, dtype=np.float32), np.array(Y_rows, dtype=np.int32)


def train_models(X_train, Y_train, X_val, Y_val, pca_dims) -> dict:
    """Fit and validate every model, returning only the optimal refit estimator and its
    chosen hyperparameters per model (not the full search/study). This is the expensive
    work that gets cached."""
    model_results = []
    best_name, best_loss = None, float("inf")

    for entry in build_models(pca_dims):
        name, search = entry["name"], entry["model"]
        print(f"\nFitting {name}...")
        search.fit(X_train, Y_train)
        model = search.best_estimator_  # discard the study / non-optimal candidates

        Y_val_pred = model.predict(X_val)
        Y_val_proba = positive_proba(model, X_val)
        per_country = [float(accuracy_score(Y_val[:, i], Y_val_pred[:, i])) for i in range(len(COUNTRIES))]

        model_results.append({
            "name": name,
            "model": model,
            "best_params": search.best_params_,
            "per_country": per_country,
            "exact": float(accuracy_score(Y_val, Y_val_pred)),
            "loss": mean_cross_entropy(Y_val, Y_val_proba),
        })

        if model_results[-1]["loss"] < best_loss:
            best_loss, best_name = model_results[-1]["loss"], name

    return {"models": model_results, "best_name": best_name, "best_loss": best_loss}


def print_results(results: dict) -> None:
    for r in results["models"]:
        print(f"\nValidation results ({r['name']}):")
        if r["best_params"] is not None:
            print(f"    Best CV params: {r['best_params']}")
        for country, acc in zip(COUNTRIES, r["per_country"]):
            print(f"    {country}: {acc:.4f}")
        print(f"    Exact match: {r['exact']:.4f}")
        print(f"    Cross-entropy loss: {r['loss']:.4f}")
    print(f"\nBest model: {results['best_name']} (cross-entropy loss: {results['best_loss']:.4f})")


def _best_model(results: dict):
    return next(r["model"] for r in results["models"] if r["name"] == results["best_name"])


def load_data():
    """Load the working-paper embedding dataset filtered to the target countries."""
    print("Loading working paper embeddings...")
    getter = DocumentTextGetter()
    docs = getter.get_all_of_type("WorkingPaper", with_embeddings=True)
    print(f"  Total working papers with embeddings: {len(docs)}")

    print("Building dataset (filtering to target countries)...")
    X, Y = build_dataset(docs)
    print(f"  Papers after filtering: {X.shape[0]}")
    print(f"  Embedding dimension:    {X.shape[1]}")
    print("  Label counts per country:")
    for i, country in enumerate(COUNTRIES):
        print(f"    {country}: {int(Y[:, i].sum())}")
    return X, Y


def split_data(X, Y):
    """70 / 15 / 15 train / validation / test split (distinct seeds per split)."""
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.50, random_state=VAL_TEST_SPLIT_RANDOM_STATE
    )
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def pca_search_dims(n_train: int, n_features: int) -> list[int]:
    """Powers of two up to 4096, capped at what a CV training fold can support
    (n_components <= min(n_features, fold samples))."""
    max_components = min(n_features, (n_train * (CV_FOLDS - 1)) // CV_FOLDS)
    return [d for d in (2 ** i for i in range(13)) if d <= max_components]


def get_or_train_results() -> dict:
    """Return cached run results, training (and caching) them if no cache exists."""
    cache_path = pathlib.Path("data/authorship_models.pickle")
    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    X, Y = load_data()
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y)
    print(f"\nSplit sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    pca_dims = pca_search_dims(len(X_train), X_train.shape[1])
    print(f"PCA dimensions searched: {pca_dims}")

    results = train_models(X_train, Y_train, X_val, Y_val, pca_dims)

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nCached results to {cache_path}")
    return results


def main():
    results = get_or_train_results()
    print_results(results)
    return _best_model(results)


if __name__ == "__main__":
    main()
