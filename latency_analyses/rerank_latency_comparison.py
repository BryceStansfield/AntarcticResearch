"""Does a cross-encoder reorder the nearest working papers into different lags?

Everything else in this folder trusts cosine distance in the Qwen embedding space
to say which working paper best anticipates an instrument. This asks whether a
second, independent judge -- a ``cohere/rerank-4-pro`` cross-encoder that reads
the instrument and each paper *together* -- would order the same candidates
differently, and whether that reordering changes the latency you would report.

The design
----------
* Take every instrument carrying an ATCM year (``N_INSTRUMENTS = None``); set it
  to an int to draw that many at random instead, under a fixed seed.
* For each, take its ``N_CANDIDATES`` nearest working papers in embedding space.
  Crucially there is **no precedence filter** here -- unlike ``measure_wp_latency``
  we do not restrict to papers that predate the instrument, so a candidate's lag
  can be negative (a paper that postdates the instrument). The eligible universe
  is still the folder's usual one: an English working paper with a real Party
  author and a known meeting year, since a paper with no year has no lag.
* Score those same candidates two ways -- by cosine similarity (the default) and
  by the reranker -- giving two orderings of one fixed set.
* At each rank ``i`` (1 = best), collect the lag of whichever paper sits there
  under each ordering. If the reranker merely reproduced cosine order the two
  lag distributions at every rank would coincide; where they part is where the
  cross-encoder disagrees about which paper is the better match.

Because the candidate set is identical under both orderings, the *pooled* lags
(across all ranks) are identical too -- only their assignment to ranks differs.
The comparison is therefore rank-by-rank, not overall.

Outputs (to ``data/latencies/``):
  * ``rerank_latency_comparison.csv`` — one row per (instrument, rank): the paper
    cosine put there and the paper the reranker put there, with both lags/scores
  * ``rerank_latency_by_rank.png``    — lag distribution at each rank, cosine vs
    reranked, as paired box-and-whiskers
"""

import concurrent.futures
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from adhoc_analyses.measure_wp_topics import load_measures, load_working_papers
from latency_analyses.measure_wp_latency import OUTPUT_DIR, _has_real_party
from latency_analyses.wp_reranker import RERANK_MODEL, rerank_scores

# None means every instrument carrying an ATCM year (~833) with no sampling at
# all, which is what this runs on. An int draws that many at random instead --
# useful for a cheap smoke test -- and cannot exceed the corpus, since the sample
# is drawn without replacement.
N_INSTRUMENTS = None
N_CANDIDATES = 10
SEED = 20260723  # fixes both the instrument sample and its reproducibility

# One rerank call per instrument, each uploading N_CANDIDATES near-full documents,
# so the run is entirely I/O bound -- fan out the way OpenRouterBackend does.
# Kept well below that backend's 64 because each request here carries ~900KB.
MAX_WORKERS = 12

# The two-series categorical pair used across this folder (validated CVD ΔE 24.7).
# Cosine is the incumbent, so it takes the folder's primary blue; the reranker is
# the challenger in orange.
COSINE = "#2a78d6"
RERANK = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d9d6"


def eligible_working_papers(working_papers: list[dict]) -> list[dict]:
    """English WPs with a real Party author and a known year -- lag needs a year."""
    return [
        w for w in working_papers
        if _has_real_party(w) and not pd.isna(w["year"])
    ]


def build_candidates(measures: list[dict], papers: list[dict]) -> pd.DataFrame:
    """One row per (instrument, rank): cosine-ranked and rerank-ranked candidates.

    The candidate *set* is the ``N_CANDIDATES`` nearest papers by cosine, shared
    by both orderings; only the order within it differs.
    """
    wp_matrix = np.vstack([w["embedding"] for w in papers])  # unit-norm
    wp_years = np.array([float(w["year"]) for w in papers])
    wp_labels = np.array([w["label"] for w in papers])
    wp_texts = [w["text"] for w in papers]

    if N_INSTRUMENTS is None:
        sampled = measures
    elif N_INSTRUMENTS > len(measures):
        raise ValueError(
            f"N_INSTRUMENTS={N_INSTRUMENTS} exceeds the {len(measures)} dated "
            f"instruments in the corpus; the sample is drawn without replacement."
        )
    else:
        rng = np.random.default_rng(SEED)
        sampled = [measures[i] for i in rng.choice(len(measures), N_INSTRUMENTS, replace=False)]

    # Candidate selection is pure numpy, so do it up front; only the rerank calls
    # need the network, and those are fanned out below.
    per_instrument = []
    for measure in sampled:
        sims = wp_matrix @ np.asarray(measure["embedding"], dtype="float32")
        # Nearest N by cosine, no precedence filter (lags may be negative).
        nearest = np.argsort(sims)[::-1][:N_CANDIDATES]
        per_instrument.append((measure, sims, nearest))

    def score(item):
        measure, _, nearest = item
        return rerank_scores(measure["text"], [wp_texts[i] for i in nearest], RERANK_MODEL)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        all_scores = list(pool.map(score, per_instrument))

    rows = []
    for (measure, sims, nearest), rerank in zip(per_instrument, all_scores):
        year = float(measure["year"])

        # Two orderings of the same candidate set.
        cosine_order = list(nearest[np.argsort(sims[nearest])[::-1]])
        rerank_order = [nearest[i] for i in np.argsort(rerank)[::-1]]
        rerank_of = dict(zip(nearest, rerank))

        for rank, (c_idx, r_idx) in enumerate(zip(cosine_order, rerank_order), start=1):
            rows.append(
                {
                    "instrument": measure["label"],
                    "instrument_type": measure["instrument_type"],
                    "instrument_year": int(year),
                    "rank": rank,
                    "cosine_wp": wp_labels[c_idx],
                    "cosine_wp_year": int(wp_years[c_idx]),
                    "cosine_lag": int(year - wp_years[c_idx]),
                    "cosine_similarity": float(sims[c_idx]),
                    "rerank_wp": wp_labels[r_idx],
                    "rerank_wp_year": int(wp_years[r_idx]),
                    "rerank_lag": int(year - wp_years[r_idx]),
                    "rerank_score": float(rerank_of[r_idx]),
                }
            )
    return pd.DataFrame(rows)


def _draw_box(ax, values: np.ndarray, position: float, color: str) -> None:
    ax.boxplot(
        [values],
        positions=[position],
        widths=0.34,
        patch_artist=True,
        boxprops=dict(facecolor=(*matplotlib.colors.to_rgb(color), 0.35),
                      edgecolor=color, linewidth=1.2),
        medianprops=dict(color=INK, linewidth=1.7),
        whiskerprops=dict(color=color, linewidth=1.1),
        capprops=dict(color=color, linewidth=1.1),
        flierprops=dict(marker="o", markersize=2.6, markerfacecolor=color,
                        markeredgecolor="none", alpha=0.5),
    )


def plot_by_rank(candidates: pd.DataFrame, path: pathlib.Path) -> None:
    """Paired box-and-whiskers: lag at each rank, cosine order vs reranked order."""
    ranks = sorted(candidates["rank"].unique())
    fig, ax = plt.subplots(figsize=(13, 6.5))

    offset = 0.19
    for rank in ranks:
        panel = candidates[candidates["rank"] == rank]
        _draw_box(ax, panel["cosine_lag"].to_numpy(float), rank - offset, COSINE)
        _draw_box(ax, panel["rerank_lag"].to_numpy(float), rank + offset, RERANK)

    # Zero = the instrument's own year. Left unlabelled: the rank-1 boxes sit
    # tight against it, and the title already reads the sign for the reader.
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlim(ranks[0] - 0.6, ranks[-1] + 0.6)
    ax.set_xlabel("rank within the instrument's 10 nearest working papers (1 = best)",
                  fontsize=10, color=INK_MUTED)
    ax.set_ylabel("lag (years): instrument year − working-paper year",
                  fontsize=10, color=INK_MUTED)

    n = candidates["instrument"].nunique()
    scope = f"all {n} dated instruments" if N_INSTRUMENTS is None else f"{n} random instruments"
    ax.set_title(
        f"Lag at each rank: cosine order vs {RERANK_MODEL} reranked order\n"
        f"{scope} · {N_CANDIDATES} nearest papers each · "
        f"same candidate set, two orderings · negative = paper postdates the instrument",
        fontsize=13, color=INK, loc="left", pad=14,
    )

    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    handles = [
        Patch(facecolor=(*matplotlib.colors.to_rgb(COSINE), 0.35),
              edgecolor=COSINE, label="cosine order"),
        Patch(facecolor=(*matplotlib.colors.to_rgb(RERANK), 0.35),
              edgecolor=RERANK, label="reranked order"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10,
              labelcolor=INK_MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached embeddings...")
    measures = [m for m in load_measures() if not pd.isna(m["year"])]
    papers = eligible_working_papers(load_working_papers())
    print(f"  {len(measures)} dated instruments, {len(papers)} eligible working papers")

    scope = "all" if N_INSTRUMENTS is None else f"{N_INSTRUMENTS} random"
    print(f"\nReranking {scope} instruments × {N_CANDIDATES} candidates "
          f"with {RERANK_MODEL} ({MAX_WORKERS} workers, cached)...")
    candidates = build_candidates(measures, papers)
    candidates.to_csv(OUTPUT_DIR / "rerank_latency_comparison.csv", index=False)

    # How much did the reranker actually move things? Rank-1 agreement is the
    # sharpest single number: did the cross-encoder keep cosine's top pick?
    top = candidates[candidates["rank"] == 1]
    agree = (top["cosine_wp"] == top["rerank_wp"]).mean()
    print(f"\nrank-1 agreement: reranker kept cosine's nearest paper for "
          f"{agree:.0%} of instruments")

    print("\nmedian lag at each rank (years):")
    print(f"  {'rank':>4}{'cosine':>9}{'rerank':>9}{'Δ median':>10}")
    for rank, g in candidates.groupby("rank"):
        cm, rm = g["cosine_lag"].median(), g["rerank_lag"].median()
        print(f"  {rank:>4}{cm:>9.0f}{rm:>9.0f}{rm - cm:>10.0f}")

    # Pooled over ranks the two sets are identical by construction; the spread is
    # in how each ordering *assigns* those lags to ranks.
    print(f"\npooled lag (identical set, both orderings): "
          f"median {candidates['cosine_lag'].median():.0f}  "
          f"mean {candidates['cosine_lag'].mean():.1f}  "
          f"[{candidates['cosine_lag'].min()}, {candidates['cosine_lag'].max()}]")
    neg = (candidates["cosine_lag"] < 0).mean()
    print(f"candidates postdating their instrument (negative lag): {neg:.0%}")

    plot_by_rank(candidates, OUTPUT_DIR / "rerank_latency_by_rank.png")
    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
