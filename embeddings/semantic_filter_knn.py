"""Quantitative separability check for the semantic-importance filter: cosine kNN (k=5).

Companion to ``semantic_filter_umap`` — instead of eyeballing a projection, ask directly: do a
sentence's nearest neighbours (by cosine distance in the raw embedding space) share its
IMPORTANT/FLUFF label? A cross-validated k=5 kNN answers that. If it beats the majority-class
baseline by a clear margin, local geometry carries the label and a semi-supervised approach
(label propagation / seed-classifier) is viable.

Uses the same label+embedding equijoin as ``semantic_filter_umap`` (no re-embedding).

Caveat: sentences are deduped by hash, but near-duplicate boilerplate recurs across papers, so
CV (which splits sentences, not documents) can flatter the FLUFF class a little.
"""
import array
import pathlib
import sqlite3

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

EMBEDDINGS_DB = pathlib.Path("data/document_embeddings.sqlite3")
SEMANTIC_FILTER_DB = pathlib.Path("data/semantic_filter.sqlite3")
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
SEMANTIC_FILTER_MODEL = "openai/gpt-oss-120b"

K = 5
CV_FOLDS = 5
RANDOM_STATE = 42


def load_labelled_embeddings() -> tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(EMBEDDINGS_DB)
    conn.execute("ATTACH ? AS sf", (str(SEMANTIC_FILTER_DB),))
    rows = conn.execute(
        """
        SELECT e.embedding, s.important
        FROM embeddings e
        JOIN sf.sentence_importance s ON s.sentence_hash = e.document_uuid
        WHERE e.model_uuid = ? AND s.model = ?
        """,
        (EMBEDDING_MODEL, SEMANTIC_FILTER_MODEL),
    ).fetchall()
    conn.close()
    X = np.array([array.array("f", blob).tolist() for blob, _ in rows], dtype=np.float32)
    y = np.array([lbl for _, lbl in rows], dtype=np.int32)
    return X, y


def main() -> None:
    print("Loading labelled embeddings...")
    X, y = load_labelled_embeddings()
    n_imp, n_fluff = int((y == 1).sum()), int((y == 0).sum())
    print(f"  {len(y)} sentences: {n_imp} important / {n_fluff} fluff (fluff base rate {n_fluff/len(y):.1%})")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    clf = KNeighborsClassifier(n_neighbors=K, metric="cosine", n_jobs=-1)

    print(f"Cross-validating k={K} cosine kNN ({CV_FOLDS}-fold)...")
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=None)
    # Probabilities = fraction of the k neighbours that are IMPORTANT — for a threshold-free AUC.
    y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]

    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    print("\n================ cosine kNN (k=5) — 5-fold CV ================")
    print(f"accuracy            {acc:.4f}   (majority-class baseline {max(n_imp, n_fluff)/len(y):.4f})")
    print(f"balanced accuracy   {bal:.4f}   (baseline 0.5000)")
    print(f"ROC-AUC             {auc:.4f}   (baseline 0.5000)")
    print("\nper class:            precision   recall     f1")
    for lbl, name in ((1, "IMPORTANT"), (0, "FLUFF    ")):
        p = precision_score(y, y_pred, pos_label=lbl, zero_division=0)
        r = recall_score(y, y_pred, pos_label=lbl, zero_division=0)
        f = f1_score(y, y_pred, pos_label=lbl, zero_division=0)
        print(f"  {name}          {p:.4f}     {r:.4f}   {f:.4f}")
    print("\nconfusion matrix (rows=true [FLUFF, IMPORTANT], cols=pred [FLUFF, IMPORTANT]):")
    print(cm)


if __name__ == "__main__":
    main()
