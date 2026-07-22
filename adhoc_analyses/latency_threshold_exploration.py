"""Does backward-looking matching degenerate? A threshold exploration.

The worry this answers: if we pick the *earliest* working paper above a
similarity threshold, a loose threshold will reach further and further back
until it simply finds the oldest paper in the corpus. Because later instruments
have more history behind them, lags would then grow with the calendar purely
mechanically -- an artefact that looks exactly like "topics take longer now".

The degeneracy ceiling is explicit here: ``available_history`` is how far back a
given instrument *could* reach (its year minus the earliest eligible paper). A
matching rule whose median lag rides that ceiling has stopped measuring
anything. A useful threshold sits well below it while still covering most
instruments.

Prior art: commit 991537e considered a cosine-distance cutoff for the Annex-V
matching problem and rejected it, keeping author/date filters instead. This
looks at a different question -- not "is this match good enough to keep" but
"how far back may we look" -- so a threshold may fare differently.

Outcome: 0.85, recorded as ``SIMILARITY_THRESHOLD`` in ``measure_wp_latency.py``.
The lag bump that survives it in the 2020s is not degeneracy -- Annex-V (Fast
Approval) instruments go from 0% of instruments through the 1990s to 29% (2000s),
51% (2010s) and 58% (2020s), and those are ASPA/ASMA management-plan *revisions*
whose originating designation genuinely does sit decades earlier.

Outputs (to ``adhoc_analyses/output/``):
  * ``threshold_sweep.csv``            — the sweep table
  * ``threshold_exploration.png``      — four diagnostic panels
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adhoc_analyses.measure_wp_topics import OUTPUT_DIR, load_measures, load_working_papers
from adhoc_analyses.measure_wp_latency import SIMILARITY_THRESHOLD, _has_real_party

THRESHOLDS = np.round(np.arange(0.70, 0.951, 0.01), 2)
# The by-decade panel: one clearly-degenerate threshold, the agreed one, and one
# stricter, so the chosen value can be read against its neighbours.
HIGHLIGHT = [0.75, SIMILARITY_THRESHOLD, 0.90]

SERIES = ["#2a78d6", "#008300", "#e87ba4"]
CEILING = "#52514e"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d9d6"


def build_matrix():
    """Similarity matrix between instruments and eligible working papers."""
    measures = [m for m in load_measures() if not pd.isna(m["year"])]
    papers = load_working_papers()

    keep = [
        w for w in papers
        if _has_real_party(w) and not pd.isna(w["year"])
    ]
    wp_years = np.array([float(w["year"]) for w in keep])
    wp_matrix = np.vstack([w["embedding"] for w in keep])
    measure_matrix = np.vstack([np.asarray(m["embedding"], dtype="float32") for m in measures])
    measure_years = np.array([float(m["year"]) for m in measures])
    measure_types = np.array([m["instrument_type"] for m in measures])

    # Unit-norm vectors, so this is cosine similarity.
    sims = measure_matrix @ wp_matrix.T
    return sims, measure_years, measure_types, wp_years


def sweep(sims, measure_years, wp_years, mask=None) -> pd.DataFrame:
    """For each threshold, how far back does the earliest-match rule reach?"""
    if mask is None:
        mask = np.ones(len(measure_years), dtype=bool)

    precedes = wp_years[None, :] <= measure_years[:, None]
    # The furthest back each instrument could possibly reach.
    available = np.where(
        precedes.any(axis=1),
        measure_years - np.where(precedes, wp_years[None, :], np.inf).min(axis=1),
        np.nan,
    )

    rows = []
    for t in THRESHOLDS:
        ok = precedes & (sims >= t)
        has = ok.any(axis=1) & mask
        if not has.any():
            continue

        # earliest above threshold
        years_masked = np.where(ok, wp_years[None, :], np.inf)
        lag_earliest = measure_years - years_masked.min(axis=1)

        # nearest above threshold (the most similar surviving paper)
        sims_masked = np.where(ok, sims, -np.inf)
        nearest_idx = sims_masked.argmax(axis=1)
        lag_nearest = measure_years - wp_years[nearest_idx]

        le, ln, av = lag_earliest[has], lag_nearest[has], available[has]
        rows.append(
            {
                "threshold": t,
                "coverage": has.sum() / mask.sum(),
                "n_covered": int(has.sum()),
                "median_lag_earliest": float(np.median(le)),
                "mean_lag_earliest": float(le.mean()),
                "median_lag_nearest": float(np.median(ln)),
                "median_available_history": float(np.median(av)),
                # 1.0 means the rule always reached the oldest paper it could.
                # nanmedian: instruments from the corpus's first years have zero
                # available history, so the ratio is undefined for them.
                "degeneracy_ratio": float(np.nanmedian(np.where(av > 0, le / np.maximum(av, 1e-9), np.nan))),
                "median_candidates": float(np.median(ok[has].sum(axis=1))),
            }
        )
    return pd.DataFrame(rows)


def by_decade(sims, measure_years, wp_years) -> pd.DataFrame:
    """Median lag per instrument decade, per highlighted threshold.

    This is the direct test of the worry: if lags grow with the calendar at a
    loose threshold but stay flat at a strict one, the growth is an artefact of
    how much history was available, not a real slowdown.
    """
    precedes = wp_years[None, :] <= measure_years[:, None]
    available = np.where(
        precedes.any(axis=1),
        measure_years - np.where(precedes, wp_years[None, :], np.inf).min(axis=1),
        np.nan,
    )
    decade = (measure_years // 10 * 10).astype(int)

    rows = []
    for d in sorted(set(decade)):
        in_decade = decade == d
        row = {"decade": d, "n": int(in_decade.sum()),
               "available_history": float(np.nanmedian(available[in_decade]))}
        for t in HIGHLIGHT:
            ok = precedes & (sims >= t)
            has = ok.any(axis=1) & in_decade
            if has.any():
                lag = measure_years - np.where(ok, wp_years[None, :], np.inf).min(axis=1)
                row[f"lag_{t}"] = float(np.median(lag[has]))
                row[f"cov_{t}"] = float(has.sum() / in_decade.sum())
            else:
                row[f"lag_{t}"] = np.nan
                row[f"cov_{t}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot(sims, table, decades, path: pathlib.Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for ax in axes.flat:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    # A: where does "similar" stop meaning anything?
    ax = axes[0][0]
    ordered = np.sort(sims, axis=1)[:, ::-1]
    bins = np.linspace(0.4, 1.0, 70)
    for data, color, label in [
        (ordered[:, 0], SERIES[0], "rank-1 match"),
        (ordered[:, 4], SERIES[1], "rank-5 match"),
        (sims.ravel()[:: max(1, sims.size // 200000)], SERIES[2], "random pair (null)"),
    ]:
        ax.hist(data, bins=bins, density=True, histtype="step", linewidth=1.8, color=color, label=label)
    ax.set_title("A. Similarity: matches vs the null", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("cosine similarity", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("density", fontsize=9, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    # B: how much of the corpus survives each threshold?
    ax = axes[0][1]
    ax.plot(table["threshold"], table["coverage"] * 100, color=SERIES[0], linewidth=2)
    ax.set_title("B. Instruments keeping at least one preceding match", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("similarity threshold", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("% of instruments covered", fontsize=9, color=INK_MUTED)
    ax.set_ylim(0, 102)

    # C: does the earliest-match rule ride the ceiling?
    ax = axes[1][0]
    ax.plot(table["threshold"], table["median_available_history"], color=CEILING,
            linewidth=1.6, linestyle=(0, (4, 3)), label="available history (degeneracy ceiling)")
    ax.plot(table["threshold"], table["median_lag_earliest"], color=SERIES[0],
            linewidth=2, label="earliest above threshold")
    ax.plot(table["threshold"], table["median_lag_nearest"], color=SERIES[1],
            linewidth=2, label="nearest above threshold")
    ax.set_title("C. Median lag vs the most it could be", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("similarity threshold", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("median lag (years)", fontsize=9, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    # D: the worry itself -- do lags grow with the calendar?
    ax = axes[1][1]
    ax.plot(decades["decade"], decades["available_history"], color=CEILING,
            linewidth=1.6, linestyle=(0, (4, 3)), label="available history")
    for t, color in zip(HIGHLIGHT, SERIES):
        ax.plot(decades["decade"], decades[f"lag_{t}"], color=color, linewidth=2,
                marker="o", markersize=4, label=f"threshold {t}")
    ax.set_title("D. Median lag by instrument decade", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("instrument decade", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("median lag (years)", fontsize=9, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    fig.suptitle("Choosing a similarity threshold for backward-looking WP → instrument matching",
                 fontsize=13.5, color=INK, x=0.007, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading cached embeddings...")
    sims, measure_years, measure_types, wp_years = build_matrix()
    print(f"  {sims.shape[0]} instruments x {sims.shape[1]} eligible working papers")

    print("\nSimilarity distribution (all instrument-paper pairs = the null):")
    ordered = np.sort(sims, axis=1)[:, ::-1]
    for name, data in [("rank-1", ordered[:, 0]), ("rank-5", ordered[:, 4]), ("null", sims.ravel())]:
        qs = np.percentile(data, [5, 25, 50, 75, 95])
        print(f"  {name:<7} p5 {qs[0]:.3f}  p25 {qs[1]:.3f}  median {qs[2]:.3f}  "
              f"p75 {qs[3]:.3f}  p95 {qs[4]:.3f}")
    print(f"  null p99 {np.percentile(sims, 99):.3f}   null p99.9 {np.percentile(sims, 99.9):.3f}")

    table = sweep(sims, measure_years, wp_years)
    table.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)

    print("\nThreshold sweep (all instruments):")
    print(f"  {'thr':>5}{'cover':>8}{'cands':>7}{'lag(earliest)':>15}{'lag(nearest)':>14}"
          f"{'ceiling':>9}{'degeneracy':>12}")
    for r in table.itertuples():
        print(f"  {r.threshold:>5.2f}{r.coverage:>8.1%}{r.median_candidates:>7.0f}"
              f"{r.median_lag_earliest:>15.1f}{r.median_lag_nearest:>14.1f}"
              f"{r.median_available_history:>9.1f}{r.degeneracy_ratio:>12.2f}")

    is_measure = measure_types == "Measure"
    m_table = sweep(sims, measure_years, wp_years, mask=is_measure)
    print(f"\nSame sweep, Type == Measure only (n={int(is_measure.sum())}):")
    print(f"  {'thr':>5}{'cover':>8}{'lag(earliest)':>15}{'lag(nearest)':>14}{'degeneracy':>12}")
    for r in m_table.itertuples():
        print(f"  {r.threshold:>5.2f}{r.coverage:>8.1%}{r.median_lag_earliest:>15.1f}"
              f"{r.median_lag_nearest:>14.1f}{r.degeneracy_ratio:>12.2f}")

    decades = by_decade(sims, measure_years, wp_years)
    print("\nMedian lag by instrument decade (the degeneration test):")
    cols = ["decade", "n", "available_history"] + [f"lag_{t}" for t in HIGHLIGHT]
    print(decades[cols].to_string(index=False))

    plot(sims, table, decades, OUTPUT_DIR / "threshold_exploration.png")
    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
