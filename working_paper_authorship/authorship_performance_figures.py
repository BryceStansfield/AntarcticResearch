"""Bar charts of the working-paper authorship benchmark, ranked by binary cross-entropy.

Reads the validation reports written by country_authorship_classifier.write_report (the
full-space run and, for the comparison figure, the --orthogonalize-country run) and renders
three PNGs into data/author_classification_figures/ (git-ignored — regenerate with
``python -m working_paper_authorship.authorship_performance_figures``):

  1) raw_methods.png                 — uncensored embeddings only;
  2) censorship_methods.png          — naive and LLM censorship;
  3) censorship_vs_orthogonal.png    — both censorship families against the orthogonal
                                       (country-signal-projected-out) run.

Every figure sorts its bars by cross-entropy (lower is better) and includes the
random-guess baseline, so a method that has learned nothing sits beside it.
"""
import argparse
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FULL_REPORT = pathlib.Path("data/author_classification_models/report.txt")
ORTHOGONAL_REPORT = pathlib.Path("data/author_classification_models_orthogonal/report.txt")
FIGURES_DIR = pathlib.Path("data/author_classification_figures")

Y_LABEL = "binary cross entropy, lower is better"

# Human-readable names for the dataset slugs of country_authorship_classifier.DATASETS.
DATASET_LABELS = {
    "raw__full": "raw",
    "naive__full": "naive censorship",
    "llm_censorship__full": "LLM censorship",
}
# Categorical hues, assigned per dataset in fixed order and shared across all three figures so a
# method keeps its colour from one chart to the next. Orthogonal runs reuse the dataset's hue and
# are distinguished by hatching (composite encoding), not by a fifth set of colours.
DATASET_COLORS = {
    "raw__full": "#2a78d6",
    "naive__full": "#eb6834",
    "llm_censorship__full": "#1baf7a",
}
BASELINE_COLOR = "#8a8a85"
ORTHOGONAL_HATCH = "///"

CENSORSHIP_DATASETS = ["naive__full", "llm_censorship__full"]
ALL_DATASETS = ["raw__full", *CENSORSHIP_DATASETS]

# The report is fixed-width; anchor on the known model names so the two-word ones parse cleanly.
_MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]
_ROW_RE = re.compile(
    rf"^(?P<model>{'|'.join(_MODEL_NAMES)})\s+(?P<dataset>\S+)\s+(?P<loss>[\d.]+)\s+(?P<exact>[\d.]+)\s"
)
_BASELINE_RE = re.compile(r"^Random-guess BCE baseline[^:]*:\s+(?P<loss>[\d.]+)")


def read_report(path: pathlib.Path) -> tuple[list[dict], float]:
    """Parse one report.txt into (rows, random-guess baseline cross-entropy)."""
    baseline = None
    rows = []
    for line in path.read_text().splitlines():
        if (m := _BASELINE_RE.match(line)) is not None:
            baseline = float(m.group("loss"))
        elif (m := _ROW_RE.match(line)) is not None:
            rows.append({
                "model": m.group("model"),
                "dataset": m.group("dataset"),
                "loss": float(m.group("loss")),
            })
    if baseline is None or not rows:
        raise ValueError(f"{path} has no baseline line and/or no model rows — is it a benchmark report?")
    return rows, baseline


def select(rows: list[dict], datasets: list[str], orthogonal: bool = False) -> list[dict]:
    """Rows for the given datasets, tagged with whether they came from the orthogonal run."""
    return [{**r, "orthogonal": orthogonal} for r in rows if r["dataset"] in datasets]


def _bar_label(row: dict) -> str:
    dataset = DATASET_LABELS.get(row["dataset"], row["dataset"])
    if row["orthogonal"]:
        dataset += "\n(orthogonalized)"
    return f"{row['model']}\n{dataset}"


def render(bars: list[dict], baseline: float, title: str, subtitle: str, path: pathlib.Path) -> pathlib.Path:
    """Draw one ranked bar chart. ``bars`` are model rows; the baseline is added as its own bar so
    it sorts into position among them, and repeated as a reference line for scanning across."""
    ordered = sorted(bars, key=lambda r: r["loss"])
    labels = [_bar_label(r) for r in ordered] + ["random guess\n(class base rates)"]
    values = [r["loss"] for r in ordered] + [baseline]
    colors = [DATASET_COLORS[r["dataset"]] for r in ordered] + [BASELINE_COLOR]
    hatches = [ORTHOGONAL_HATCH if r["orthogonal"] else "" for r in ordered] + [""]

    # Tick labels are rotated multi-line strings, so horizontal room has to grow with the bar
    # count (and again when the "(orthogonalized)" third line is in play) or they collide.
    per_bar = 0.95 if any(r["orthogonal"] for r in ordered) else 0.75
    width = max(6.0, per_bar * len(values) + 1.8)
    # Grow the height with the width past a point, or the many-bar figure ends up a squashed strip.
    fig, ax = plt.subplots(figsize=(width, max(5.6, 0.33 * width)), dpi=200)
    drawn = ax.bar(range(len(values)), values, color=colors, width=0.78,
                   edgecolor="white", linewidth=1.0)
    for bar, hatch in zip(drawn, hatches):
        bar.set_hatch(hatch)
    # Every bar carries its value: the aqua/yellow hues sit under 3:1 against a white surface, so
    # the numbers (and the legend) are what make each bar identifiable, not the fill alone.
    ax.bar_label(drawn, fmt="%.3f", padding=2, fontsize=7, color="#52514e")

    ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.0, zorder=0)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel(Y_LABEL, fontsize=9)
    ax.set_ylim(0, max(values) * 1.16)
    ax.set_title(title, fontsize=11, loc="left", pad=22)
    # Offset in points, not axes fractions: the figure height varies with the bar count, and an
    # axes-fraction offset would drift into the title on the tall figures.
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", fontsize=8, color="#52514e", va="bottom")

    ax.grid(axis="y", color="#dedcd6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dedcd6")
    ax.tick_params(axis="both", length=0, labelsize=8, colors="#52514e")

    handles = [Patch(facecolor=DATASET_COLORS[d], edgecolor="white", label=DATASET_LABELS[d])
               for d in ALL_DATASETS if any(r["dataset"] == d for r in ordered)]
    if any(r["orthogonal"] for r in ordered):
        handles.append(Patch(facecolor=BASELINE_COLOR, edgecolor="white", hatch=ORTHOGONAL_HATCH,
                             label="orthogonalized (country signal projected out)"))
    handles.append(Patch(facecolor=BASELINE_COLOR, edgecolor="white", label="random-guess baseline"))
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left", ncols=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all_figures(full_report: pathlib.Path = FULL_REPORT,
                       orthogonal_report: pathlib.Path = ORTHOGONAL_REPORT,
                       figures_dir: pathlib.Path = FIGURES_DIR) -> list[pathlib.Path]:
    """Render whichever of the three figures the available reports support."""
    if not full_report.exists():
        print(f"No benchmark report at {full_report} — skipping figures.")
        return []
    rows, baseline = read_report(full_report)

    written = [
        render(select(rows, ["raw__full"]), baseline,
               "Authorship classification on uncensored working papers",
               "Validation set; ranked by binary cross-entropy.",
               figures_dir / "raw_methods.png"),
        render(select(rows, CENSORSHIP_DATASETS), baseline,
               "Authorship classification under naive and LLM censorship",
               "Validation set; ranked by binary cross-entropy.",
               figures_dir / "censorship_methods.png"),
    ]

    if orthogonal_report.exists():
        orth_rows, orth_baseline = read_report(orthogonal_report)
        if abs(orth_baseline - baseline) > 1e-4:
            # Both runs score the same validation split, so the no-skill reference must agree;
            # if it doesn't, the two reports are from different data and are not comparable.
            raise ValueError(
                f"Baseline mismatch: {full_report} reports {baseline:.4f} but {orthogonal_report} "
                f"reports {orth_baseline:.4f} — the reports describe different validation sets."
            )
        written.append(render(
            select(rows, CENSORSHIP_DATASETS) + select(orth_rows, ALL_DATASETS, orthogonal=True),
            baseline,
            "Censorship methods vs. orthogonal decomposition of the country signal",
            "Validation set; ranked by binary cross-entropy.",
            figures_dir / "censorship_vs_orthogonal.png"))
    else:
        print(f"No orthogonal report at {orthogonal_report} — skipping the comparison figure "
              f"(run the benchmark with --orthogonalize-country to produce it).")

    for path in written:
        print(f"Wrote {path}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full-report", type=pathlib.Path, default=FULL_REPORT)
    parser.add_argument("--orthogonal-report", type=pathlib.Path, default=ORTHOGONAL_REPORT)
    parser.add_argument("--figures-dir", type=pathlib.Path, default=FIGURES_DIR)
    args = parser.parse_args()
    render_all_figures(args.full_report, args.orthogonal_report, args.figures_dir)


if __name__ == "__main__":
    main()
