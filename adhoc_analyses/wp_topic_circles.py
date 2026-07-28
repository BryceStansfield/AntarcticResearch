"""Circle-packing figure of the working-paper-only BERTopic topics.

One circle per topic from the model in ``topic_introduction.get_wp_bertopic()``, with area
proportional to how many working papers the topic holds, packed into a 1920x1080 canvas.
Circles big enough to hold it carry their topic's top c-TF-IDF keyword; the HDBSCAN outlier
topic (-1, the papers that fell into no cluster) is red and labelled "NO TOPIC".

Renders to data/topic_figures/wp_topic_circles.png (git-ignored):
    uv run python -m adhoc_analyses.wp_topic_circles
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

OUTPUT_PATH = pathlib.Path("data/topic_figures/wp_topic_circles.png")

WIDTH, HEIGHT, DPI = 1920, 1080, 100
BACKGROUND = "#ffffff"
TOPIC_COLOR = "#2a78d6"
OUTLIER_COLOR = "#e34948"
LABEL_COLOR = "#ffffff"
OUTLIER_LABEL = "NO TOPIC"

# Fraction of the canvas the circles' combined area should cover. Relaxation packing of
# mixed-radius circles tops out around 0.7; below that the gaps grow visibly.
AREA_FILL = 0.74
# Kept as clear canvas between neighbouring circles (the surface gap that keeps two same-colour
# fills from reading as one shape) and around the canvas edge.
CIRCLE_GAP = 4.0
MARGIN = 6.0

# A packing is accepted once no two circles are closer than CIRCLE_GAP minus this slack.
OVERLAP_TOLERANCE = 0.1
# If relaxation cannot separate everything inside the canvas, every radius is scaled by this and
# the packing retried. A uniform scale leaves the count-to-area proportionality untouched.
RETRY_SHRINK = 0.985
MAX_PACKING_ATTEMPTS = 12

MIN_FONT_SIZE = 7.0
MAX_FONT_SIZE = 40.0
# Longest chord a label may occupy, and the tallest it may stand, as fractions of the diameter.
LABEL_WIDTH_FRACTION = 0.82
LABEL_HEIGHT_FRACTION = 0.42


def topic_sizes() -> list[dict]:
    """(topic id, working-paper count, top keyword) for every topic in the WP-only model."""
    from antarctic_ladder_metrics.topic_introduction import get_wp_bertopic

    info = get_wp_bertopic().topic_model.get_topic_info()
    topics = []
    for row in info.itertuples():
        representation = list(row.Representation)
        topics.append({
            "topic": int(row.Topic),
            "count": int(row.Count),
            "keyword": OUTLIER_LABEL if row.Topic == -1 else (representation[0] if representation else ""),
            "outlier": row.Topic == -1,
        })
    return topics


def radii_for(counts: np.ndarray, width: int, height: int, fill: float) -> np.ndarray:
    """Radii with area proportional to count, scaled so the circles cover ``fill`` of the canvas."""
    scale = np.sqrt(fill * width * height / (np.pi * counts.sum()))
    return scale * np.sqrt(counts)


def pack(radii: np.ndarray, width: int, height: int, iterations: int = 3000,
         settle_iterations: int = 2000, gap: float = CIRCLE_GAP,
         margin: float = MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """Pack circles into the rectangle by relaxation, largest first.

    Seeds positions on a golden-angle spiral (so the big circles start near the middle and the
    small ones fill outwards), then repeatedly pushes overlapping pairs apart and nudges
    everything back toward the centre, clamping each circle inside the canvas. Displacement is
    split between a pair in inverse proportion to area, so a small circle gives way to a large
    one instead of shoving it across the canvas.

    The first ``iterations`` steps compact the layout under a fading centre-ward pull; the
    ``settle_iterations`` that follow run with the pull switched off and stop as soon as no
    circles overlap, so compaction can never win out over separation at the end.
    """
    n = len(radii)
    order = np.argsort(-radii)
    r = radii[order]

    i = np.arange(n)
    angle = i * np.pi * (3.0 - np.sqrt(5.0))
    spread = np.sqrt((i + 0.5) / n)
    x = width / 2 + spread * (width / 2 - margin) * np.cos(angle)
    y = height / 2 + spread * (height / 2 - margin) * np.sin(angle)

    # Inverse-area weights: w_ij is the share of a pair's overlap that circle i absorbs.
    area = r ** 2
    weight = area[None, :] / (area[:, None] + area[None, :])
    min_dist = r[:, None] + r[None, :] + gap
    np.fill_diagonal(min_dist, 0.0)

    rng = np.random.default_rng(42)
    for step in range(iterations + settle_iterations):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        # Coincident centres have no separating direction; jitter them apart.
        coincident = (dist < 1e-9) & (min_dist > 0)
        if coincident.any():
            dx = np.where(coincident, rng.normal(size=dx.shape), dx)
            dy = np.where(coincident, rng.normal(size=dy.shape), dy)
            dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-9)

        overlap = np.maximum(min_dist - dist, 0.0)
        np.fill_diagonal(overlap, 0.0)
        push = weight * overlap / np.maximum(dist, 1e-9)
        x += (push * dx).sum(axis=1)
        y += (push * dy).sum(axis=1)

        # Gravity toward the centre closes the gaps the pushes open up; it fades out over the
        # compaction phase and is off entirely once settling starts.
        if step < iterations:
            pull = 0.02 * (1.0 - step / iterations)
            x += pull * (width / 2 - x)
            y += pull * (height / 2 - y)

        np.clip(x, r + margin, width - r - margin, out=x)
        np.clip(y, r + margin, height - r - margin, out=y)

        if step >= iterations and overlap.max(initial=0.0) <= OVERLAP_TOLERANCE:
            break

    unsorted_x, unsorted_y = np.empty(n), np.empty(n)
    unsorted_x[order] = x
    unsorted_y[order] = y
    return unsorted_x, unsorted_y


def max_overlap(x: np.ndarray, y: np.ndarray, radii: np.ndarray) -> float:
    """Largest pairwise overlap in pixels; 0 means the packing is clean."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = radii[:, None] + radii[None, :] - dist
    np.fill_diagonal(overlap, 0.0)
    return float(overlap.max())


def pack_to_fit(radii: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack, shrinking every radius uniformly until nothing overlaps. Returns (x, y, radii)."""
    for attempt in range(MAX_PACKING_ATTEMPTS):
        x, y = pack(radii, width, height)
        if max_overlap(x, y, radii) <= OVERLAP_TOLERANCE:
            return x, y, radii
        radii = radii * RETRY_SHRINK
    print(f"Warning: {max_overlap(x, y, radii):.2f}px of overlap remains after "
          f"{MAX_PACKING_ATTEMPTS} packing attempts — lower AREA_FILL.")
    return x, y, radii


_FONT = FontProperties(weight="bold")


def _text_extent(text: str) -> tuple[float, float]:
    """Width and height of ``text`` at font size 1, in points."""
    box = TextPath((0, 0), text, size=1.0, prop=_FONT).get_extents()
    return box.width, box.height


def fit_font_size(text: str, radius: float) -> float | None:
    """Largest font size at which ``text`` fits inside the circle, or None if it never does."""
    if not text:
        return None
    unit_width, unit_height = _text_extent(text)
    if unit_width <= 0 or unit_height <= 0:
        return None
    points_per_pixel = 72.0 / DPI
    by_width = (LABEL_WIDTH_FRACTION * 2 * radius / unit_width) * points_per_pixel
    by_height = (LABEL_HEIGHT_FRACTION * 2 * radius / unit_height) * points_per_pixel
    size = min(by_width, by_height, MAX_FONT_SIZE)
    return size if size >= MIN_FONT_SIZE else None


def render(topics: list[dict], path: pathlib.Path = OUTPUT_PATH) -> pathlib.Path:
    counts = np.array([t["count"] for t in topics], dtype=float)
    radii = radii_for(counts, WIDTH, HEIGHT, AREA_FILL)
    x, y, radii = pack_to_fit(radii, WIDTH, HEIGHT)

    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)

    labelled = 0
    for topic, cx, cy, radius in zip(topics, x, y, radii):
        color = OUTLIER_COLOR if topic["outlier"] else TOPIC_COLOR
        ax.add_patch(plt.Circle((cx, cy), radius, facecolor=color, edgecolor="none"))
        font_size = fit_font_size(topic["keyword"], radius)
        if font_size is not None:
            labelled += 1
            ax.text(cx, cy, topic["keyword"], ha="center", va="center",
                    fontsize=font_size, color=LABEL_COLOR, fontweight="bold")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"{len(topics)} topics, {int(counts.sum())} working papers; {labelled} circles labelled; "
          f"max overlap {max_overlap(x, y, radii):.2f}px")
    print(f"Wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    render(topic_sizes(), args.output)


if __name__ == "__main__":
    main()
