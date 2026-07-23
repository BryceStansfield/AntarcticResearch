"""Distribution of working-paper -> instrument lag, by decade and instrument type.

``measure_wp_latency.py`` reports lag summarised per *topic*; this asks a
different cut of the same matched lags: how does the lag distribution differ
between instrument types, and how has it moved over the decades? Two orientations
of the one dataset, each a grid of box-and-whisker plots:

  * ``lag_box_by_decade.png`` — one panel per decade, a box per instrument type
  * ``lag_box_by_type.png``   — one panel per instrument type, a box per decade

Box plots (not densities) because the lags are small integers with a long right
tail: the median, the quartile box and the whiskers say plainly what a kernel
density blurs, and a handful of matches still makes an honest box where it makes a
misleading curve.

Both figures share one lag axis, capped just above the bulk of the data
(``AXIS_PERCENTILE``) so the boxes are legible rather than crushed against a
scale stretched by a few extreme lags; the rare points above the cap fall outside
the view. The per-box count printed above each box says how many instruments it
rests on.

The lags come from ``measure_wp_latency.match_instruments`` -- the earliest
preceding working paper above ``SIMILARITY_THRESHOLD`` -- so, like everything in
this folder, latency is >= 0 by construction and the caveats in
``measure_wp_latency.py`` (type-driven unmatched bias especially) apply here too.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch

from adhoc_analyses.measure_wp_topics import load_measures, load_working_papers
from latency_analyses.measure_wp_latency import OUTPUT_DIR, match_instruments

# Fixed colour per instrument type, applied identically in every panel so a type
# keeps its identity across the small multiples. Order is the categorical order
# (blue, orange, aqua, yellow). Untyped is left out entirely -- it matched too few
# instruments to summarise. Colour is redundant here (position and the legend both
# name the group), so it carries no accessibility weight of its own.
TYPE_COLOR = {
    "Measure":        "#2a78d6",
    "Recommendation": "#eb6834",
    "Resolution":     "#1baf7a",
    "Decision":       "#eda100",
}

# A box over one or two points shows a spread that isn't there; require a few
# before drawing one, and skip a type/decade cell with fewer.
MIN_FOR_BOX = 3

# The lag axis is capped at this percentile of all matched lags so the boxes get
# the vertical room the long tail would otherwise steal. Points above the cap sit
# outside the view rather than flattening every box near the floor.
AXIS_PERCENTILE = 98

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d9d6"


def matched_lags():
    """Every matched instrument's lag, with its type and decade.

    Loads once and matches over *all* instruments (no type filter) -- the panels
    are the type breakdown, so the matching must not pre-select a type.
    """
    measures = load_measures()
    working_papers = load_working_papers()
    rows = match_instruments(measures, working_papers)
    matched = rows[rows["matched"]].copy()
    matched["decade"] = (matched["instrument_year"] // 10 * 10).astype(int)
    # Untyped is dropped folder-wide from these figures.
    return matched[matched["instrument_type"].isin(TYPE_COLOR)].copy()


def _draw_box(ax, values: np.ndarray, position: int, color: str) -> None:
    """A single coloured box-and-whisker at ``position`` (tinted fill, inked median)."""
    ax.boxplot(
        [values],
        positions=[position],
        widths=0.62,
        patch_artist=True,
        boxprops=dict(facecolor=to_rgba(color, 0.35), edgecolor=color, linewidth=1.2),
        medianprops=dict(color=INK, linewidth=1.7),
        whiskerprops=dict(color=color, linewidth=1.1),
        capprops=dict(color=color, linewidth=1.1),
        flierprops=dict(marker="o", markersize=2.6, markerfacecolor=color,
                        markeredgecolor="none", alpha=0.5),
    )


def _style_panel(ax, title: str, n_positions: int, ymax: float) -> None:
    """Shared panel chrome: title, y-grid, recessive spines, no category ticks."""
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=8)
    ax.set_xticks([])  # identity comes from colour + the shared legend
    ax.set_xlim(0.5, n_positions + 0.5)
    ax.set_ylim(0, ymax)


def _count_label(ax, position: int, n: int) -> None:
    """Print a box's sample size just under the top of the panel."""
    ax.text(position, 0.98, str(n), transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color=INK_MUTED)


def _finish(fig, axes, all_axes, n_visible, handles, suptitle, path):
    """Hide spare cells, restore x-column ticks, add labels/legend/title, save."""
    ncols = axes.shape[1]
    nrows = axes.shape[0]
    for ax in all_axes[n_visible:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("lag (years)", fontsize=9, color=INK_MUTED)

    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK_MUTED,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(suptitle, fontsize=13.5, color=INK, x=0.007, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_box_by_decade(matched, path: pathlib.Path) -> int:
    """One panel per decade; one box-and-whisker per instrument type within each."""
    decades = sorted(matched["decade"].unique())
    if not decades:
        return 0
    types = list(TYPE_COLOR)
    ymax = float(np.percentile(matched["latency_years"], AXIS_PERCENTILE)) * 1.03

    ncols = min(4, len(decades))
    nrows = int(np.ceil(len(decades) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    all_axes = list(axes.flat)

    drawn = 0
    for ax, decade in zip(all_axes, decades):
        panel = matched[matched["decade"] == decade]
        for pos, instrument_type in enumerate(types, start=1):
            vals = panel.loc[panel["instrument_type"] == instrument_type,
                             "latency_years"].to_numpy(dtype=float)
            if len(vals) < MIN_FOR_BOX:
                continue
            _draw_box(ax, vals, pos, TYPE_COLOR[instrument_type])
            _count_label(ax, pos, len(vals))
            drawn += 1
        _style_panel(ax, f"{decade}s  (n={len(panel)})", len(types), ymax)

    handles = [Patch(facecolor=to_rgba(c, 0.35), edgecolor=c, label=t)
               for t, c in TYPE_COLOR.items()]
    _finish(fig, axes, all_axes, len(decades), handles,
            "Working paper → instrument lag: by instrument type, per decade", path)
    return drawn


def plot_box_by_type(matched, path: pathlib.Path) -> int:
    """One panel per instrument type; one box-and-whisker per decade within each."""
    decades = sorted(matched["decade"].unique())
    if not decades:
        return 0
    types = list(TYPE_COLOR)
    ymax = float(np.percentile(matched["latency_years"], AXIS_PERCENTILE)) * 1.03

    # Ordered gradient over the decades (a decade keeps its colour across panels),
    # truncated short of plasma's palest yellow so every decade reads on white.
    base = plt.get_cmap("plasma")
    cmap = LinearSegmentedColormap.from_list("decade", base(np.linspace(0.06, 0.85, 256)))
    norm = plt.Normalize(min(decades), max(decades))
    decade_color = {d: cmap(norm(d)) for d in decades}

    ncols = 2
    nrows = int(np.ceil(len(types) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.8 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    all_axes = list(axes.flat)

    drawn = 0
    for ax, instrument_type in zip(all_axes, types):
        panel = matched[matched["instrument_type"] == instrument_type]
        for pos, decade in enumerate(decades, start=1):
            vals = panel.loc[panel["decade"] == decade,
                             "latency_years"].to_numpy(dtype=float)
            if len(vals) < MIN_FOR_BOX:
                continue
            _draw_box(ax, vals, pos, decade_color[decade])
            _count_label(ax, pos, len(vals))
            drawn += 1
        _style_panel(ax, f"{instrument_type}  (n={len(panel)})", len(decades), ymax)

    handles = [Patch(facecolor=to_rgba(decade_color[d], 0.55),
                     edgecolor=decade_color[d], label=f"{d}s") for d in decades]
    _finish(fig, axes, all_axes, len(types), handles,
            "Working paper → instrument lag: by decade, per instrument type", path)
    return drawn


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached embeddings and matching instruments...")
    matched = matched_lags()
    print(f"  {len(matched)} matched instruments across "
          f"{matched['decade'].nunique()} decades")

    print("\nmatched instruments by decade x type:")
    table = (matched.pivot_table(index="decade", columns="instrument_type",
                                 values="latency_years", aggfunc="size", fill_value=0))
    print(table.to_string())

    n_by_decade = plot_box_by_decade(matched, OUTPUT_DIR / "lag_box_by_decade.png")
    n_by_type = plot_box_by_type(matched, OUTPUT_DIR / "lag_box_by_type.png")
    print(f"\n{n_by_decade} boxes in the by-decade figure, {n_by_type} in the by-type "
          f"figure (>= {MIN_FOR_BOX} matches per box)")
    print(f"Written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
