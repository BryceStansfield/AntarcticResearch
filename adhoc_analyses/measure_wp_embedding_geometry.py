"""Where do the not-yet-effective measures sit in the Qwen embedding space, relative to WPs?

Motivated by the failure of the working-paper authorship classifiers to predict measure authorship
(see ``working_paper_authorship/not_effective_measure_authorship.py``): is that failure because the
measures land in an unseen region of the embedding space (an OOD-coverage problem), or because they
sit *inside* the working-paper manifold and the failure is purely about the authorship signal (a
decision-boundary problem)?

To tell those apart we compare three cosine-distance distributions over whole-document raw
embeddings (same Qwen embedder for both classes):
  1) WP - WP           — spread within the working-paper corpus
  2) measure - measure  — spread within the 52 not-yet-effective measures
  3) WP - measure      — how far measures sit from working papers
plus, per measure, the distance to its *nearest* working paper — the sharpest test of whether
measures have close WP neighbours or float off on their own.

Like the other adhoc analyses this reads the Qwen vectors already cached in
``data/document_embeddings.sqlite3`` and never touches the network. WP vectors come from the cached
``WorkingPaper`` type; measure vectors are pulled by recomputing each measure's exact segment hash
(the uncensored ``Content``), so we score exactly the 52 measures the classifier predicted on.

Outputs (to ``adhoc_analyses/output/``):
  * ``measure_wp_geometry_report.txt``   — the distribution summary + interpretation-ready numbers
  * ``measure_wp_geometry_summary.csv``  — one row per distribution, the same statistics
  * ``measure_wp_geometry_hist.png``     — overlaid histograms of the three distributions
"""
import itertools
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from embeddings.document_embeddings import (
    get_embeddings_by_type, get_embedding, get_wp_ip_embedding_args,
)

OUTPUT_DIR = pathlib.Path("adhoc_analyses/output")
MEASURES_CSV = pathlib.Path("data/Not-Effective measures.csv")
WP_EMBEDDING_TYPE = "WorkingPaper"
MAX_WITHIN_WP_PAIRS = 400_000  # subsample the ~3.2M WP-WP pairs; the distribution is stable well below this
RANDOM_SEED = 0


def unit_rows(mat: np.ndarray) -> np.ndarray:
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def load_wp_vectors() -> np.ndarray:
    """Cached whole-document raw working-paper embeddings, unit-normalised."""
    pairs = get_embeddings_by_type(WP_EMBEDDING_TYPE)
    return unit_rows(np.asarray([v for _, v in pairs], dtype=np.float64))


def load_measure_vectors() -> np.ndarray:
    """The 52 measures' whole-document raw embeddings, fetched by recomputing their segment hashes.

    Uses the same hashing the classifier used (``get_wp_ip_embedding_args`` over the raw ``Content``),
    so these are exactly the vectors that were fed to the authorship models. Any measure whose
    embedding is missing from the cache is skipped with a warning rather than silently zero-filled."""
    df = pd.read_csv(MEASURES_CSV)
    vectors, missing = [], 0
    for row in df.itertuples():
        content = getattr(row, "Content", None)
        if pd.isna(content) or not str(content).strip():
            continue
        for h, _t, _seg in get_wp_ip_embedding_args(str(content), "measure_geometry"):
            embedding = get_embedding(h)
            if embedding is None:
                missing += 1
                continue
            vectors.append(embedding)
    if missing:
        print(f"  warning: {missing} measure segment(s) had no cached embedding and were skipped "
              f"(run the measure embedding step first).")
    return unit_rows(np.asarray(vectors, dtype=np.float64))


def within_distances(mat: np.ndarray, rng: np.random.Generator, max_pairs: int) -> np.ndarray:
    """Upper-triangle cosine distances within one set, subsampling pairs if there are too many."""
    n = mat.shape[0]
    pairs = list(itertools.combinations(range(n), 2))
    if len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in chosen]
    i = np.fromiter((a for a, _ in pairs), dtype=int, count=len(pairs))
    j = np.fromiter((b for _, b in pairs), dtype=int, count=len(pairs))
    return 1.0 - np.einsum("kd,kd->k", mat[i], mat[j])


def cross_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (1.0 - a @ b.T).ravel()


def summarise(distances: np.ndarray) -> dict:
    p05, p50, p95 = np.percentile(distances, [5, 50, 95])
    return {
        "n_pairs": int(distances.size),
        "mean": float(distances.mean()),
        "std": float(distances.std()),
        "min": float(distances.min()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "max": float(distances.max()),
    }


def build_report(rows: list[tuple[str, dict]], nn: np.ndarray, within_wp_mean: float,
                 n_wp: int, n_measure: int) -> str:
    lines = [
        "MEASURE vs WORKING-PAPER EMBEDDING GEOMETRY",
        "Whole-document raw Qwen embeddings; cosine distance = 1 - cosine similarity.",
        f"Working papers: {n_wp}   Measures: {n_measure}",
        "",
        f"{'distribution':22s} {'n_pairs':>10s} {'mean':>7s} {'std':>7s} "
        f"{'min':>7s} {'p05':>7s} {'p50':>7s} {'p95':>7s} {'max':>7s}",
    ]
    for name, s in rows:
        lines.append(
            f"{name:22s} {s['n_pairs']:10d} {s['mean']:7.4f} {s['std']:7.4f} "
            f"{s['min']:7.4f} {s['p05']:7.4f} {s['p50']:7.4f} {s['p95']:7.4f} {s['max']:7.4f}"
        )
    lines += [
        "",
        f"Each measure's NEAREST working paper (cosine dist): "
        f"mean {nn.mean():.4f}  min {nn.min():.4f}  max {nn.max():.4f}",
        f"For reference, the mean within-WP distance is {within_wp_mean:.4f}.",
        "",
        "Reading: if the cross (WP-measure) distribution matches the within-WP distribution and every",
        "measure has a close WP neighbour (nearest-WP distance well below the typical WP-WP gap), the",
        "measures sit INSIDE the working-paper manifold. Then the authorship classifier's failure is a",
        "decision-boundary / label problem (the authorial signal is absent from boilerplate measure",
        "text), NOT an OOD-coverage problem (the embedder is not sending measures somewhere unseen).",
    ]
    return "\n".join(lines)


def plot_histograms(dists: dict[str, np.ndarray], path: pathlib.Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, d in dists.items():
        plt.hist(d, bins=60, range=(0, 1), density=True, histtype="step", linewidth=1.6, label=name.strip())
    plt.xlabel("cosine distance (1 - cosine similarity)")
    plt.ylabel("density")
    plt.title("Measure vs Working-Paper embedding geometry")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    WP = load_wp_vectors()
    ME = load_measure_vectors()
    print(f"WPs: {WP.shape[0]} vectors   Measures: {ME.shape[0]} vectors")

    dists = {
        "1) WP - WP          ": within_distances(WP, rng, MAX_WITHIN_WP_PAIRS),
        "2) measure - measure": within_distances(ME, rng, MAX_WITHIN_WP_PAIRS),
        "3) WP - measure     ": cross_distances(WP, ME),
    }
    rows = [(name, summarise(d)) for name, d in dists.items()]
    nn = (1.0 - ME @ WP.T).min(axis=1)  # each measure's nearest-WP cosine distance
    within_wp_mean = dists["1) WP - WP          "].mean()

    report = build_report(rows, nn, within_wp_mean, WP.shape[0], ME.shape[0])
    print("\n" + report)

    report_path = OUTPUT_DIR / "measure_wp_geometry_report.txt"
    report_path.write_text(report)

    summary_df = pd.DataFrame([{"distribution": name.strip(), **s} for name, s in rows])
    summary_df["nearest_wp_mean"] = [None, None, float(nn.mean())]  # only meaningful for the cross view
    summary_path = OUTPUT_DIR / "measure_wp_geometry_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    hist_path = OUTPUT_DIR / "measure_wp_geometry_hist.png"
    plot_histograms(dists, hist_path)

    print(f"\nWrote:\n  {report_path}\n  {summary_path}\n  {hist_path}")


if __name__ == "__main__":
    main()
