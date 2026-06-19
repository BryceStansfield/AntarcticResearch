"""Train and cache the differentiable RFF-SVM authorship model on its own.

This mirrors the SVM slot of country_authorship_classifier (same PCA -> clf pipeline,
Optuna search over cross-entropy, validation reporting and caching) but swaps the rbf
SVC for the autograd-friendly RFFSVM, so the result can be used to walk the embedding
hypersphere toward the document that maximises the joint authorship probability
p_1 * ... * p_n. The original classifier module is left untouched and its reusable
helpers (data loading, splitting, scoring, metrics) are imported rather than copied.
"""
import pathlib
import pickle

from optuna.distributions import FloatDistribution, CategoricalDistribution
from optuna_integration import OptunaSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from working_paper_authorship import country_authorship_classifier as cc
from working_paper_authorship.torch_svm import RFFSVM

CACHE_PATH = pathlib.Path("data/differentiable_svm.pickle")


def build_model(pca_dims: list[int]) -> OptunaSearchCV:
    """PCA -> RFFSVM pipeline with Bayesian search over the kernel/regularisation knobs."""
    pipeline = Pipeline([
        ("pca", PCA(random_state=cc.RANDOM_STATE)),
        ("clf", RFFSVM(random_state=cc.RANDOM_STATE)),
    ])
    return OptunaSearchCV(
        pipeline,
        param_distributions={
            "pca__n_components": CategoricalDistribution(pca_dims),
            "clf__C": FloatDistribution(1e-2, 1e2, log=True),
            "clf__gamma": FloatDistribution(1e-3, 1e0, log=True),
            "clf__n_features": CategoricalDistribution([256, 512, 1024]),
        },
        n_trials=cc.N_OPTUNA_TRIALS,
        scoring=cc.neg_cross_entropy_scorer,
        cv=cc.CV_FOLDS,
        n_jobs=-1,
        random_state=cc.OPTUNA_RANDOM_STATE,
    )


def train(X_train, Y_train, X_val, Y_val, pca_dims) -> dict:
    """Fit the search and return the optimal refit estimator plus validation metrics."""
    search = build_model(pca_dims)
    print("Fitting differentiable RFF-SVM...")
    search.fit(X_train, Y_train)
    model = search.best_estimator_

    Y_val_pred = model.predict(X_val)
    Y_val_proba = cc.positive_proba(model, X_val)
    per_country = [float(accuracy_score(Y_val[:, i], Y_val_pred[:, i])) for i in range(len(cc.COUNTRIES))]

    return {
        "name": "Differentiable RFF-SVM",
        "model": model,
        "best_params": search.best_params_,
        "per_country": per_country,
        "exact": float(accuracy_score(Y_val, Y_val_pred)),
        "loss": cc.mean_cross_entropy(Y_val, Y_val_proba),
    }


def get_or_train() -> dict:
    """Return the cached differentiable SVM, training (and caching) it if absent."""
    if CACHE_PATH.exists():
        print(f"Loading cached differentiable SVM from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    X, Y = cc.load_data()
    X_train, X_val, X_test, Y_train, Y_val, Y_test = cc.split_data(X, Y)
    pca_dims = cc.pca_search_dims(len(X_train), X_train.shape[1])
    print(f"PCA dimensions searched: {pca_dims}")

    result = train(X_train, Y_train, X_val, Y_val, pca_dims)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached differentiable SVM to {CACHE_PATH}")
    return result


def main() -> None:
    result = get_or_train()
    print(f"\n{result['name']}  (best params: {result['best_params']})")
    for country, acc in zip(cc.COUNTRIES, result["per_country"]):
        print(f"    {country}: {acc:.4f}")
    print(f"    Exact match: {result['exact']:.4f}")
    print(f"    Cross-entropy loss: {result['loss']:.4f}")


if __name__ == "__main__":
    main()
