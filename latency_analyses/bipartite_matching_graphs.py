"""Bipartite pictures of WP → instrument matching, one per matching policy.

The rest of the folder argues about *which* matching rule to trust with numbers
and box plots. This draws the rules instead: a bipartite graph with working
papers down the left, instruments down the right, and an edge wherever a policy
would pair them. Seeing the edge set makes the trade-offs tangible -- a loose
threshold is a visible thicket, a strict one a sparse scatter, and the backward
filter's job is literally to erase every edge that points the wrong way in time.

No edges are ever pruned, and every node is drawn whether or not it matched, so a
picture is the whole policy, hairball and all.

The policies drawn (one image each)
-----------------------------------
* **Threshold, backward-looking** -- edge iff cosine ≥ t *and* the paper predates
  (or shares the year of) the instrument. This is the family
  ``measure_wp_latency`` chooses from; every edge is non-anticipatory.
* **Threshold, unrestricted** -- edge iff cosine ≥ t, with no date filter, so a
  paper may postdate its instrument. The same graph as above plus exactly the
  edges the backward filter removes.
  Both are drawn for every t in ``THRESHOLDS`` ({0.75 … 0.90}).
* **Closest document** -- each instrument joined to its single nearest paper by
  cosine, no threshold and no date filter: the "nearest paper overall" rule
  ``measure_wp_latency`` rejected, shown so its structure can be compared.

Reading the picture
-------------------
Both columns are **stratified by year into bands, oldest at the top**, separated by
dotted lines; the two sides share one year axis, so a working paper and an
instrument of the same year sit at the same height. An edge is therefore
**horizontal within a year, sloping down** when the paper is older than its
instrument (**blue** -- a defensible match) and **sloping up** when the paper is
newer (**orange** -- it postdates the instrument, impossible as an anticipation,
the failure the backward filter exists to remove). Backward-looking graphs are
all-blue by construction; the unrestricted and closest graphs show how much
orange the date filter was hiding.

The band layout is identical across every image (it is built from all nodes, not
a policy's subset), so the graphs can be read side by side. The title reports how
many instruments matched and how many papers were used.

Outputs (to ``data/latencies/bipartite_graphs/``):
  * ``threshold_<t>_backward.png`` / ``threshold_<t>_unrestricted.png`` per t
  * ``closest_document.png``
  * ``policy_summary.csv`` — edge/coverage counts per policy
"""

import collections
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from adhoc_analyses.measure_wp_topics import load_measures, load_working_papers
from latency_analyses.measure_wp_latency import (OUTPUT_DIR, _has_real_party,
                                                 argmax_tiebroken, label_order)

# Step 0.05, endpoints as requested. A constant so the family is easy to widen.
THRESHOLDS = [0.75, 0.80, 0.85, 0.90]
OUTPUT_SUBDIR = OUTPUT_DIR / "bipartite_graphs"

# A year's band gets vertical room proportional to how many nodes it must hold,
# capped so one crowded modern year cannot dwarf every early one; sparse years
# still get a floor of room. Uniform-per-year was illegible where 100+ papers
# share a year, count-proportional made the figure metres tall -- this is the
# middle path.
BAND_UNIT_CAP = 40

# Folder palette (validated CVD ΔE 24.7). Blue = defensible (paper predates or
# shares the year), orange = the paper postdates its instrument.
VALID = "#2a78d6"
POSTDATES = "#eb6834"
NODE = "#52514e"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#c4c4c1"


def load_sides():
    """Eligible instruments and working papers, with labels, years and vectors.

    The eligible sets mirror ``latency_threshold_exploration.build_matrix`` exactly
    -- dated instruments, and English WPs with a real Party author and a year --
    so the edges here are the same pairs those analyses reason about.
    """
    instruments = [m for m in load_measures() if not pd.isna(m["year"])]
    papers = [
        w for w in load_working_papers()
        if _has_real_party(w) and not pd.isna(w["year"])
    ]

    instr = {
        "labels": np.array([m["label"] for m in instruments]),
        "years": np.array([int(m["year"]) for m in instruments]),
        "emb": np.vstack([np.asarray(m["embedding"], dtype="float32") for m in instruments]),
    }
    wp = {
        "labels": np.array([w["label"] for w in papers]),
        "years": np.array([int(w["year"]) for w in papers]),
        "emb": np.vstack([w["embedding"] for w in papers]),
    }
    return instr, wp


def edges_for_policy(sims, instr_years, wp_years, *, wp_order=None, threshold=None,
                     backward=False, closest=False):
    """(instrument_index, wp_index) pairs a policy would draw.

    ``closest`` ignores ``threshold``/``backward`` and joins each instrument to
    its single nearest paper; ``wp_order`` breaks ties there by label rather than
    by column position (see ``measure_wp_latency.label_order``).
    """
    if closest:
        if wp_order is None:
            wp_order = np.arange(sims.shape[1])
        nearest = argmax_tiebroken(sims, wp_order)
        return np.column_stack([np.arange(len(instr_years)), nearest])

    mask = sims >= threshold
    if backward:
        mask &= wp_years[None, :] <= instr_years[:, None]
    return np.argwhere(mask)  # rows of (instrument_index, wp_index)


def build_bands(instr_years, wp_years):
    """Shared per-year vertical bands in [0, 1], oldest at the top.

    Every year present on either side gets a band whose height is proportional to
    the larger of its two node counts (capped). Returns ``(years, bands)`` where
    ``bands[year] = (y_top, y_bottom)`` with ``y_top > y_bottom``.
    """
    ci = collections.Counter(instr_years.tolist())
    cw = collections.Counter(wp_years.tolist())
    years = sorted(set(ci) | set(cw))
    units = {y: min(BAND_UNIT_CAP, max(1, ci.get(y, 0), cw.get(y, 0))) for y in years}
    total = sum(units.values())

    bands, cum = {}, 0.0
    for y in years:  # ascending -> oldest first -> top
        frac = units[y] / total
        bands[y] = (1.0 - cum, 1.0 - (cum + frac))
        cum += frac
    return years, bands


def node_y(years_arr, labels_arr, bands) -> dict[int, float]:
    """A y in [0, 1] for every node, packed within its year's band.

    Nodes sharing a year are spread evenly across the band (a small margin keeps
    them off the dotted separators), ordered by label so the layout is stable.
    """
    pos = {}
    by_year = collections.defaultdict(list)
    for idx in range(len(years_arr)):
        by_year[int(years_arr[idx])].append(idx)

    for year, idxs in by_year.items():
        idxs.sort(key=lambda i: labels_arr[i])
        y_top, y_bottom = bands[year]
        margin = (y_top - y_bottom) * 0.12
        hi, lo = y_top - margin, y_bottom + margin
        ys = [(hi + lo) / 2] if len(idxs) == 1 else np.linspace(hi, lo, len(idxs))
        for idx, y in zip(idxs, ys):
            pos[idx] = float(y)
    return pos


def draw_graph(edges, instr, wp, instr_y, wp_y, years, bands, height,
               title: str, path: pathlib.Path, *, all_valid: bool) -> dict:
    """Render one policy's bipartite graph and return its summary counts."""
    fig, ax = plt.subplots(figsize=(10, height))

    # Dotted separators between year strata, spanning both columns, plus a year
    # label at the left of each band (skipped when a band is too thin to label
    # without colliding with its neighbour).
    min_label_gap = 0.16 / height
    last_label = np.inf
    boundaries = sorted({bands[y][0] for y in years} | {bands[years[-1]][1]}, reverse=True)
    for b in boundaries:
        ax.plot([-0.28, 1.28], [b, b], color=GRID, linewidth=0.7,
                linestyle=(0, (1, 2)), zorder=1)
    for year in years:
        y_top, y_bottom = bands[year]
        center = (y_top + y_bottom) / 2
        if last_label - center >= min_label_gap:
            ax.text(-0.34, center, str(year), ha="right", va="center",
                    fontsize=6.5, color=INK_MUTED)
            last_label = center

    # Edges: orange (postdating) sorted last so failures sit atop the valid mass.
    segments, is_post = [], []
    for i, w in edges:
        postdates = wp["years"][w] > instr["years"][i]
        segments.append([(0.0, wp_y[w]), (1.0, instr_y[i])])
        is_post.append(bool(postdates))
    is_post = np.array(is_post, dtype=bool)
    order = np.argsort(is_post)  # False (valid) first, True (postdating) on top
    colors = np.where(is_post, POSTDATES, VALID)
    alpha = 0.3 if len(segments) > 3000 else 0.55
    ax.add_collection(LineCollection([segments[i] for i in order],
                                     colors=colors[order], linewidths=0.5,
                                     alpha=alpha, zorder=2))

    # Every node, matched or not (no pruning).
    ax.scatter(np.zeros(len(wp_y)), list(wp_y.values()), s=5, color=NODE,
               zorder=3, edgecolors="none")
    ax.scatter(np.ones(len(instr_y)), list(instr_y.values()), s=5, color=NODE,
               zorder=3, edgecolors="none")

    ax.text(0.0, 1.028, "Working papers", ha="center", va="bottom",
            fontsize=11, color=INK, weight="bold")
    ax.text(1.0, 1.028, "Instruments", ha="center", va="bottom",
            fontsize=11, color=INK, weight="bold")

    handles = [Line2D([0], [0], color=VALID, lw=2,
                      label="paper predates / same year (edge slopes down)")]
    if all_valid:
        handles.append(Line2D([0], [0], color="none",
                              label="backward-looking: no postdating edges"))
    else:
        handles.append(Line2D([0], [0], color=POSTDATES, lw=2,
                              label="paper postdates instrument (edge slopes up)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.05),
              frameon=False, fontsize=9, labelcolor=INK_MUTED, ncol=2,
              handletextpad=0.6, columnspacing=1.8)

    ax.set_title(title, fontsize=12.5, color=INK, loc="center", pad=40)
    ax.set_xlim(-0.45, 1.32)
    ax.set_ylim(-0.02, 1.06)
    ax.axis("off")

    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    n_post = int(is_post.sum())
    return {
        "n_edges": len(edges),
        "n_instruments_matched": int(len(np.unique(edges[:, 0]))),
        "n_wps_used": int(len(np.unique(edges[:, 1]))),
        "n_postdating_edges": n_post,
        "pct_postdating": float(n_post / len(edges)) if len(edges) else 0.0,
    }


def main():
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached embeddings...")
    instr, wp = load_sides()
    print(f"  {len(instr['labels'])} dated instruments × {len(wp['labels'])} eligible working papers")

    # Shared layout, built once from all nodes so every graph aligns.
    years, bands = build_bands(instr["years"], wp["years"])
    instr_y = node_y(instr["years"], instr["labels"], bands)
    wp_y = node_y(wp["years"], wp["labels"], bands)
    # Tall enough that the busiest band's nodes separate; capped so it stays a
    # figure rather than a scroll. Driven by node counts, so it is honest about
    # how crowded the corpus is.
    height = float(np.clip(0.11 * sum(min(BAND_UNIT_CAP,
                   max(1, (instr["years"] == y).sum(), (wp["years"] == y).sum()))
                   for y in years), 12, 60))
    print(f"  {len(years)} year strata ({years[0]}–{years[-1]}), figure height {height:.0f} in")

    # Unit-norm vectors, so the dot product is cosine similarity.
    sims = instr["emb"] @ wp["emb"].T

    rows = []
    print("\nPolicy                          edges  instr  wps  postdating")
    for t in THRESHOLDS:
        for backward in (True, False):
            kind = "backward" if backward else "unrestricted"
            edges = edges_for_policy(sims, instr["years"], wp["years"],
                                     threshold=t, backward=backward)
            name = f"threshold_{t:.2f}_{kind}"
            summary = draw_graph(edges, instr, wp, instr_y, wp_y, years, bands, height,
                                 f"cosine ≥ {t:.2f}, {kind} — WP → instrument matching",
                                 OUTPUT_SUBDIR / f"{name}.png", all_valid=backward)
            rows.append({"policy": name, "threshold": t, "backward": backward, **summary})
            print(f"  {name:<28}{summary['n_edges']:>7}{summary['n_instruments_matched']:>7}"
                  f"{summary['n_wps_used']:>6}{summary['pct_postdating']:>10.1%}")

    edges = edges_for_policy(sims, instr["years"], wp["years"],
                             wp_order=label_order(wp["labels"]), closest=True)
    summary = draw_graph(edges, instr, wp, instr_y, wp_y, years, bands, height,
                         "closest document (nearest paper overall) — WP → instrument matching",
                         OUTPUT_SUBDIR / "closest_document.png", all_valid=False)
    rows.append({"policy": "closest_document", "threshold": None, "backward": False, **summary})
    print(f"  {'closest_document':<28}{summary['n_edges']:>7}{summary['n_instruments_matched']:>7}"
          f"{summary['n_wps_used']:>6}{summary['pct_postdating']:>10.1%}")

    pd.DataFrame(rows).to_csv(OUTPUT_SUBDIR / "policy_summary.csv", index=False)
    print(f"\n{len(rows)} graphs written to {OUTPUT_SUBDIR}/")


from utils import line_buffer_stdout

if __name__ == "__main__":
    line_buffer_stdout()
    main()
