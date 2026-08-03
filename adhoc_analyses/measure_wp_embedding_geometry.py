"""Where do ATCM instruments sit in the Qwen embedding space, relative to working papers?

Motivated by the failure of the working-paper authorship classifiers to predict instrument authorship
— every model scored far worse than its no-skill baseline (BCE 1.46-2.01 against a 0.30 base-rate
floor), and a topic-conditioned prior over subject matter alone did no better (AUC 0.41, p = 0.39).
Is that failure because the measures land in an unseen region of the embedding space (an
OOD-coverage problem), or because they sit *inside* the working-paper manifold and the failure is
purely about the authorship signal (a decision-boundary problem)?

The two modules that produced those numbers (``not_effective_measure_authorship.py`` and
``topic_prior_authorship.py``) have since been deleted — the negative result held up and there was
no reason to keep re-running them.

To tell those apart we compare three cosine-distance distributions over whole-document
embeddings (same Qwen embedder for both classes):
  1) WP - WP            — spread within the working-paper corpus
  2) measure - measure  — spread within the ATCM instrument corpus
  3) WP - measure       — how far instruments sit from working papers
plus, per instrument, the distance to its *nearest* working paper — the sharpest test of whether
instruments have close WP neighbours or float off on their own.

Both sides are read from vectors ``embed_all_documents`` has already cached — the ``WorkingPaper``
and ``measure`` types — so this never touches the network.

It used to run over the 52 rows of ``Not-Effective measures.csv`` instead, matching the exact set
the deleted authorship classifier was scored on. With that classifier gone the subset was
inherited rather than chosen, and it was a poor population regardless: only 3 of the 52 were
genuinely "Not yet effective". All 52 are inside this corpus. (Old method: git history.)

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

from embeddings.document_embeddings import DocumentTextGetter

OUTPUT_DIR = pathlib.Path("adhoc_analyses/output")
WP_EMBEDDING_TYPE = "WorkingPaper"
# Every ATCM instrument (Measure/Recommendation/Resolution/Decision) is stored under this one
# type by embed_all_documents.
MEASURE_EMBEDDING_TYPE = "measure"
MAX_WITHIN_WP_PAIRS = 400_000  # subsample the ~3.2M WP-WP pairs; the distribution is stable well below this
RANDOM_SEED = 0


def unit_rows(mat: np.ndarray) -> np.ndarray:
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def load_wp_vectors() -> np.ndarray:
    """Cached whole-document raw working-paper embeddings, unit-normalised.

    One vector per paper. Enumerating the embedding rows directly gives one per *segment*, so the
    handful of papers long enough to be split entered every distribution below several times over
    -- and, being segments, entered them as partial documents rather than the whole-document
    vectors this comparison is about. ``get_all_of_type`` pools a paper's segments back into the
    single vector the classifier would see.

    English only, matching the corpus every other analysis here runs on.
    """
    documents = DocumentTextGetter().get_all_of_type(WP_EMBEDDING_TYPE, with_embeddings=True)
    english = [d for d in documents if str(d.get("paper_language", "")).lower() == "english"]
    return unit_rows(np.asarray([d["embedding"] for d in english], dtype=np.float64))


def load_measure_vectors() -> np.ndarray:
    """Every embedded ATCM instrument's whole-document embedding, one vector per instrument.

    The whole corpus, not a subset. This used to read the 52 rows of ``Not-Effective measures.csv``,
    which was the right population when the module existed to explain why the authorship classifier
    failed on exactly those 52 -- the geometry had to speak to that specific failure. That
    classifier has been deleted, so the subset became inherited rather than chosen, and it was never
    a coherent category anyway: only 3 of the 52 were "Not yet effective", the other 49 having been
    withdrawn, spent, terminated or superseded.

    All 52 were a subset of this corpus, so nothing is lost and the sample grows about sixteenfold,
    at no embedding cost -- these are already cached under the ``measure`` type. The vectors are the
    Subject+Content representation the rest of the pipeline uses, rather than the raw ``Content``
    the deleted classifier was fed.
    """
    documents = DocumentTextGetter().get_all_of_type(MEASURE_EMBEDDING_TYPE, with_embeddings=True)
    if not documents:
        raise RuntimeError(
            f"No '{MEASURE_EMBEDDING_TYPE}' embeddings are cached — run embed_all_documents first. "
            f"Refusing to report distributions computed over nothing."
        )
    return unit_rows(np.asarray([d["embedding"] for d in documents], dtype=np.float64))


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
    # Range taken from the data rather than hardcoded to (0, 1). Cosine distance runs to 2, and
    # nothing here guarantees non-negative similarity -- a hardcoded range silently drops any pair
    # outside it, and dropping the very pairs that would show measures sitting *opposite* the
    # working-paper manifold would quietly flatter the conclusion this figure is drawn to test.
    # A shared range across the three series is what keeps them comparable.
    lo = min(float(d.min()) for d in dists.values())
    hi = max(float(d.max()) for d in dists.values())
    for name, d in dists.items():
        plt.hist(d, bins=60, range=(lo, hi), density=True, histtype="step", linewidth=1.6, label=name.strip())
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


from utils import line_buffer_stdout

if __name__ == "__main__":
    line_buffer_stdout()
    main()
