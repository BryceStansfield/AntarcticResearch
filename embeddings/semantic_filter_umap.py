"""Data experiment: UMAP the censored-WP sentence embeddings, coloured by semantic importance.

Question we're probing: could the (currently per-sentence LLM) semantic-relevance filter of
``working_paper_semantic_filter`` instead be learned in a semi-supervised setting? If IMPORTANT
and FLUFF sentences already separate in the embedding space, a small labelled seed + the geometry
might suffice.

What it does:
  1) Pull every censored WP sentence that has BOTH a cached importance label
     (``data/semantic_filter.sqlite3``) AND a cached embedding (``data/document_embeddings.sqlite3``).
     The importance cache keys on ``sha256(sentence)`` and the embedder keys each unit on
     ``sha256(text)`` too, so an equijoin on ``sentence_hash == document_uuid`` pairs each label
     with the embedding of the byte-identical sentence — no re-embedding, no ambiguity.
  2) UMAP the embeddings down to 2D and to 3D (PCA-50 pre-reduction for speed/stability).
  3) Scatter each projection coloured by IMPORTANT vs FLUFF (static PNGs + an interactive 3D HTML).

Coverage: once ``embed_all_llm_censored_working_paper_sentences`` (in ``embed_all_documents``) has
embedded every paper's LLM-censored sentences, the join covers ~all labelled sentences (128,798 of
128,801; the 3 missing are long sentences that ``split_long_document`` sub-split, so they're stored
under sub-segment hashes rather than ``sha256(whole sentence)``). Earlier runs saw only ~52% — the
target-country subset the authorship classifier had embedded — so re-run the embed pass first.
"""
import array
import pathlib
import sqlite3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap

EMBEDDINGS_DB = pathlib.Path("data/document_embeddings.sqlite3")
SEMANTIC_FILTER_DB = pathlib.Path("data/semantic_filter.sqlite3")
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
SEMANTIC_FILTER_MODEL = "openai/gpt-oss-120b"

OUTPUT_DIR = pathlib.Path("data/semantic_filter_umap")
RANDOM_STATE = 42
PCA_COMPONENTS = 50  # pre-reduce 4096-d embeddings before UMAP (standard recipe: faster, denoises)

# Colourblind-safe pair (Okabe-Ito): important = blue, fluff = orange.
COLOR_IMPORTANT = "#0072B2"
COLOR_FLUFF = "#E69F00"


def load_labelled_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X is (n, 4096) float32 embeddings, y is (n,) int (1=IMPORTANT, 0=FLUFF),
    for every sentence that has both a cached label and a cached embedding."""
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


def run_umap(X: np.ndarray, n_components: int) -> np.ndarray:
    """PCA-50 -> UMAP to ``n_components`` dims (cosine metric, fixed seed for reproducibility)."""
    pcs = PCA(n_components=min(PCA_COMPONENTS, X.shape[1]), random_state=RANDOM_STATE).fit_transform(X)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=True,
    )
    return reducer.fit_transform(pcs)


def _shuffled(n: int) -> np.ndarray:
    """A fixed-seed permutation so neither class is systematically drawn on top of the other."""
    rng = np.random.default_rng(RANDOM_STATE)
    order = np.arange(n)
    rng.shuffle(order)
    return order


def plot_2d(emb: np.ndarray, y: np.ndarray, path: pathlib.Path) -> None:
    order = _shuffled(len(y))
    colors = np.where(y == 1, COLOR_IMPORTANT, COLOR_FLUFF)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(emb[order, 0], emb[order, 1], c=colors[order], s=2, alpha=0.35, linewidths=0)
    ax.set_title("Censored WP sentences — UMAP (2D), coloured by semantic importance")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    _legend(ax, y)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3d_static(emb: np.ndarray, y: np.ndarray, path: pathlib.Path) -> None:
    order = _shuffled(len(y))
    colors = np.where(y == 1, COLOR_IMPORTANT, COLOR_FLUFF)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(emb[order, 0], emb[order, 1], emb[order, 2],
               c=colors[order], s=2, alpha=0.35, linewidths=0)
    ax.set_title("Censored WP sentences — UMAP (3D), coloured by semantic importance")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    _legend(ax, y)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3d_interactive(emb: np.ndarray, y: np.ndarray, path: pathlib.Path) -> None:
    fig = go.Figure()
    for label, name, color in ((0, "FLUFF", COLOR_FLUFF), (1, "IMPORTANT", COLOR_IMPORTANT)):
        m = y == label
        fig.add_trace(go.Scatter3d(
            x=emb[m, 0], y=emb[m, 1], z=emb[m, 2],
            mode="markers", name=f"{name} ({int(m.sum())})",
            marker=dict(size=1.6, color=color, opacity=0.5),
        ))
    fig.update_layout(
        title="Censored WP sentences — UMAP (3D), coloured by semantic importance",
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(path)


def _legend(ax, y: np.ndarray) -> None:
    n_imp, n_fluff = int((y == 1).sum()), int((y == 0).sum())
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_IMPORTANT, label=f"IMPORTANT ({n_imp})"),
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_FLUFF, label=f"FLUFF ({n_fluff})"),
    ]
    ax.legend(handles=handles, loc="best")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading labelled embeddings...")
    X, y = load_labelled_embeddings()
    print(f"  {len(y)} sentences: {int((y == 1).sum())} important / {int((y == 0).sum())} fluff, dim={X.shape[1]}")

    print("UMAP -> 2D...")
    emb2d = run_umap(X, 2)
    print("UMAP -> 3D...")
    emb3d = run_umap(X, 3)

    # Persist coordinates so replotting/relabelling doesn't recompute UMAP.
    np.savez_compressed(OUTPUT_DIR / "umap_coords.npz", emb2d=emb2d, emb3d=emb3d, y=y)

    print("Plotting...")
    plot_2d(emb2d, y, OUTPUT_DIR / "umap_2d.png")
    plot_3d_static(emb3d, y, OUTPUT_DIR / "umap_3d.png")
    plot_3d_interactive(emb3d, y, OUTPUT_DIR / "umap_3d_interactive.html")
    print(f"Wrote plots + coords to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
