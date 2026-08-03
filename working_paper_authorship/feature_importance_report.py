"""Probe what each authorship classifier keys on in the PCA'd embedding space.

For every cached model — one per (model, dataset) pair trained by country_authorship_classifier —
we rank its PCA components by feature importance (native where available: forests'
impurity/gain importances and logistic-regression coefficients; otherwise permutation
importance), then for the top components show the segments that project most strongly positive
and negative along them.

Segments are that dataset's *own* units (whole censored documents, or kept sentences), rebuilt
through the classifier's own chunking so the text projected matches the text the PCA was fitted
on. Run country_authorship_classifier first — this reads its pickles and embedding cache, and
never trains. Output is written to data/EmbeddingFeatureReport.txt.
"""
import json
import pathlib
import pickle

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.inspection import permutation_importance

from working_paper_authorship import country_authorship_classifier as cc

REPORT_PATH = pathlib.Path("data/EmbeddingFeatureReport.txt")
TOP_COMPONENTS = 5
TOP_SEGMENTS = 3
SNIPPET_CHARS = 1000
PERMUTATION_RANDOM_STATE = 99


def component_importances(pipeline, X_val, Y_val) -> tuple[np.ndarray, str]:
    """Importance of each PCA component for one fitted PCA -> classifier pipeline.

    Uses a native importance where the classifier exposes one, otherwise falls back to
    permutation importance computed in the PCA-transformed feature space."""
    pca = pipeline.named_steps["pca"]
    clf = pipeline.named_steps["clf"]

    # Native: tree ensembles (RandomForest, XGBoost) expose impurity/gain importances.
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_), "native (impurity/gain feature_importances_)"

    # Native: linear models expose per-label coefficients — aggregate |coef| over labels.
    #
    # Scaled by each component's standard deviation, because a coefficient alone is not an
    # importance when the features are not on a common scale. What a component contributes to the
    # logit is |coef| * sd: PCA components are uncorrelated but explicitly *not* standardised —
    # their variances are the eigenvalues, and here they span orders of magnitude — so a large
    # coefficient on a component that barely varies moves the prediction less than a small one on a
    # component that varies a lot. Ranking on |coef| alone is biased toward the low-variance tail,
    # which is precisely the direction that would manufacture support for the "classifiers key on
    # fine non-topic directions" hypothesis this report exists to test. On
    # logistic_regression__raw__full it put components [16, 13, 28, 30, 22] on top while the true
    # top contributors are [1, 0, 6, 16, 13] — the two highest-variance, most plausibly topical
    # components never appeared at all. The tree and permutation branches are already scale-aware,
    # so this also keeps the three methods measuring the same thing.
    if isinstance(clf, MultiOutputClassifier) and all(hasattr(e, "coef_") for e in clf.estimators_):
        coefs = np.vstack([np.abs(e.coef_).ravel() for e in clf.estimators_])
        component_sd = np.sqrt(pca.explained_variance_)
        return (coefs.mean(axis=0) * component_sd,
                "native (mean |coef| x component sd across country labels)")

    # Fallback (e.g. rbf SVM): permutation importance, scored by cross-entropy.
    transformed = pca.transform(X_val)
    result = permutation_importance(
        clf, transformed, Y_val,
        scoring=cc.neg_cross_entropy_scorer,
        random_state=PERMUTATION_RANDOM_STATE,
    )
    return result.importances_mean, "permutation importance"


def load_dataset_segments(records: list[dict], method: str):
    """Rebuild one dataset's units over `records`, as (X, Y, texts, stems).

    Goes through the classifier's own dataset_units/assemble_xy, so the segments are exactly the
    ones the dataset's PCA was fitted on — same censorship, same chunking, same hashes. ``texts``
    stays row-aligned with X because assemble_xy raises on a missing embedding rather than
    silently dropping the row."""
    embed_units, hash_labels = cc.dataset_units(records, method)
    by_hash = {h: text for h, _type, text in embed_units}
    X, Y, stems = cc.assemble_xy(hash_labels)
    texts = [by_hash[h] for h, _label, _stem in hash_labels]
    return X, Y, texts, stems


def format_meta(stem: str) -> str:
    return f"paper={stem}"


def format_snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > SNIPPET_CHARS:
        collapsed = collapsed[:SNIPPET_CHARS] + "…"
    return "          " + collapsed


def build_report() -> None:
    records = cc.load_working_papers()
    train, val, _test = cc.split_documents(records)
    # Segments are drawn from train + val, giving the component exploration a wide pool to find
    # extremes in. The test split is left out so this report can be rerun freely without eroding it.
    projection_records = train + val
    val_stems = {r["stem"] for r in val}

    lines: list[str] = [
        "WORKING PAPER AUTHORSHIP — EMBEDDING FEATURE REPORT",
        f"Countries: {', '.join(cc.COUNTRIES)}",
        f"Segments projected per dataset: train + val documents ({len(projection_records)} papers)",
        f"Top components per model: {TOP_COMPONENTS} | extreme segments per component: {TOP_SEGMENTS}",
        "Importances and cross-entropy are computed on the validation split alone.",
        "",
    ]

    for slug, censorship in cc.DATASETS:
        # Which models exist for this dataset. Checked before rebuilding segments, because that
        # re-runs censorship — expensive work to do for a dataset the benchmark has not trained
        # yet (and assemble_xy would then raise on its missing embeddings anyway).
        trained = [n for n in cc.MODEL_NAMES
                   if (cc.OUTPUT_DIR / f"{cc.model_slug(n)}__{slug}.pickle").exists()]
        if not trained:
            print(f"Skipping {slug}: no trained models in {cc.OUTPUT_DIR}/ — run the benchmark first.")
            continue
        hp_path = cc.OUTPUT_DIR / cc.HYPERPARAMETERS_FILENAME
        # One shared search across datasets, so these are the same params for every slug.
        best_params = cc.load_shared_hyperparameters(cc.OUTPUT_DIR) if hp_path.exists() else {}

        print(f"Rebuilding segments for {slug}...")
        # Built once over train + val: the projection pool. The validation rows used for
        # importances and loss are then masked out of it, rather than rebuilt — rebuilding
        # re-runs censorship, so doing it twice would be needless expense.
        X_proj, Y_proj, texts, stems = load_dataset_segments(projection_records, censorship)
        val_mask = np.array([s in val_stems for s in stems])
        X_val, Y_val = X_proj[val_mask], Y_proj[val_mask]

        for name in trained:
            with open(cc.OUTPUT_DIR / f"{cc.model_slug(name)}__{slug}.pickle", "rb") as f:
                pipeline = pickle.load(f)
            pca = pipeline.named_steps["pca"]

            print(f"  computing importances for {name} on {slug}...")
            importances, importance_method = component_importances(pipeline, X_val, Y_val)
            n_show = min(TOP_COMPONENTS, importances.shape[0])
            top_components = np.argsort(importances)[::-1][:n_show]

            projection = pca.transform(X_proj)  # (n_segments, n_components)
            loss = cc.mean_cross_entropy(Y_val, cc.positive_proba(pipeline, X_val))

            lines.append("=" * 80)
            lines.append(f"MODEL: {name}  |  DATASET: {slug}")
            lines.append(f"Segments projected: {len(texts)}")
            lines.append(f"Validation cross-entropy: {loss:.4f} | best params: {best_params.get(name, '[unknown]')}")
            lines.append(f"PCA components retained: {pca.n_components_}")
            lines.append(f"Importance method: {importance_method}")
            lines.append("")

            for k in top_components:
                column = projection[:, k]
                order = np.argsort(column)
                lines.append(f"  PCA component #{int(k)}  (importance {importances[k]:.5f})")
                for header, idxs in (("HIGHEST-projecting segments", order[::-1][:TOP_SEGMENTS]),
                                     ("LOWEST-projecting segments", order[:TOP_SEGMENTS])):
                    lines.append(f"    --- {header} ---")
                    for idx in idxs:
                        lines.append(f"      [{column[idx]:+.3f}] {format_meta(stems[idx])}")
                        lines.append(format_snippet(texts[idx]))
                lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH} ({len(lines)} lines)")


from utils import line_buffer_stdout

if __name__ == "__main__":
    line_buffer_stdout()
    build_report()
