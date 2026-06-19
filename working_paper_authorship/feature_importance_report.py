"""Probe what each authorship classifier keys on in the PCA'd embedding space.

For every cached best model we rank its PCA components by feature importance (native
where available — forests' impurity/gain importances and logistic-regression
coefficients — otherwise permutation importance), then for the top components show the
working-paper segments that project most strongly positive and negative along them.
Output is written to data/EmbeddingFeatureReport.txt.
"""
import pathlib
import hashlib

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.inspection import permutation_importance

import embeddings.document_embeddings as de
from embeddings.document_embeddings import DocumentTextGetter, get_embeddings_by_type
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
    if isinstance(clf, MultiOutputClassifier) and all(hasattr(e, "coef_") for e in clf.estimators_):
        coefs = np.vstack([np.abs(e.coef_).ravel() for e in clf.estimators_])
        return coefs.mean(axis=0), "native (mean |coef| across country labels)"

    # Fallback (e.g. rbf SVM): permutation importance, scored by cross-entropy.
    transformed = pca.transform(X_val)
    result = permutation_importance(
        clf, transformed, Y_val,
        scoring=cc.neg_cross_entropy_scorer,
        random_state=PERMUTATION_RANDOM_STATE,
    )
    return result.importances_mean, f"permutation importance"


def load_segment_embeddings() -> tuple[list[str], np.ndarray]:
    """All working-paper segment embeddings, as (uuids, matrix)."""
    pairs = get_embeddings_by_type("WorkingPaper")
    uuids = [uuid for uuid, _ in pairs]
    matrix = np.asarray([embedding for _, embedding in pairs], dtype=np.float32)
    return uuids, matrix


def segment_details(getter: DocumentTextGetter, uuid: str) -> tuple[dict, str]:
    """Return (metadata, exact-segment-text) for a segment uuid. The embedding uuid is a
    hash of one segment, so we re-split the source document and match it back."""
    if uuid not in getter.wp_ip_map:
        return {}, "[segment text unavailable]"
    rep = getter.get_document_representation(uuid)
    text = rep.get("text", "")
    for segment in de.split_long_document(text):
        if hashlib.sha256(segment.encode()).hexdigest() == uuid:
            return rep, segment
    return rep, text  # single-segment document, or hashing mismatch


def format_meta(uuid: str, rep: dict) -> str:
    bits = [f"uuid={uuid[:12]}"]
    if rep.get("sort_string"):
        bits.append(str(rep["sort_string"]))
    if rep.get("parties") is not None:
        bits.append(f"parties={rep['parties']}")
    if rep.get("paper_language"):
        bits.append(f"lang={rep['paper_language']}")
    return " | ".join(bits)


def format_snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > SNIPPET_CHARS:
        collapsed = collapsed[:SNIPPET_CHARS] + "…"
    return "          " + collapsed


def build_report() -> None:
    results = cc.get_or_train_results()

    # The validation split is needed for permutation importance (the SVM path).
    X, Y = cc.load_data()
    _, X_val, _, _, Y_val, _ = cc.split_data(X, Y)

    getter = DocumentTextGetter()
    uuids, embeddings = load_segment_embeddings()

    lines: list[str] = [
        "WORKING PAPER AUTHORSHIP — EMBEDDING FEATURE REPORT",
        f"Countries: {', '.join(cc.COUNTRIES)}",
        f"Working-paper segments projected: {len(uuids)}",
        f"Top components per model: {TOP_COMPONENTS} | extreme segments per component: {TOP_SEGMENTS}",
        "",
        "PCA dimensions retained per model:",
    ]
    for r in results["models"]:
        lines.append(f"  {r['name']}: {r['model'].named_steps['pca'].n_components_}")
    lines.append("")

    for r in results["models"]:
        name, pipeline = r["name"], r["model"]
        pca = pipeline.named_steps["pca"]

        print(f"Computing importances for {name}...")
        importances, method = component_importances(pipeline, X_val, Y_val)
        n_show = min(TOP_COMPONENTS, importances.shape[0])
        top_components = np.argsort(importances)[::-1][:n_show]

        projection = pca.transform(embeddings)  # (n_segments, n_components)

        lines.append("=" * 80)
        lines.append(f"MODEL: {name}")
        lines.append(f"Validation cross-entropy: {r['loss']:.4f} | best params: {r['best_params']}")
        lines.append(f"PCA components retained: {pca.n_components_}")
        lines.append(f"Importance method: {method}")
        lines.append("")

        for k in top_components:
            column = projection[:, k]
            order = np.argsort(column)
            lines.append(f"  PCA component #{int(k)}  (importance {importances[k]:.5f})")
            for header, idxs in (("HIGHEST-projecting segments", order[::-1][:TOP_SEGMENTS]),
                                 ("LOWEST-projecting segments", order[:TOP_SEGMENTS])):
                lines.append(f"    --- {header} ---")
                for idx in idxs:
                    uuid = uuids[idx]
                    rep, segment = segment_details(getter, uuid)
                    lines.append(f"      [{column[idx]:+.3f}] {format_meta(uuid, rep)}")
                    lines.append(format_snippet(segment))
            lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    build_report()
