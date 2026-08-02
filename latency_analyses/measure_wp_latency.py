"""Working-Paper -> instrument latency, matched per instrument rather than per topic.

``measure_wp_topics.py`` answers "what is in each topic". This answers "how long
did each instrument take to arrive after the working paper that anticipated it",
by matching every ATCM instrument to a single working paper and summarising the
resulting latencies per topic.

Matching each instrument individually means the latency does not inherit the
breadth of its topic -- the topic is only the grouping used for reporting.

The matching rule
-----------------
For each instrument, take the **earliest working paper that predates it and
scores at least ``SIMILARITY_THRESHOLD``**, breaking ties on the earliest year by
similarity. This estimates when text this similar first appeared, which is the
closest defensible proxy for when the idea surfaced.

Two earlier rules were tried and rejected:

* *Nearest paper overall.* Dominated by near-duplicate text -- a sibling revision
  of the same ASPA plan, or the instrument's own draft tabled at the same
  meeting. 24.5% of matches postdated their instrument and only 31.8% preceded
  it.
* *Nearest 5 papers.* Widening found a preceding paper for 80.8% of instruments,
  but a random eligible paper would have managed 87.3% -- below chance, because
  rank-5 matches (median similarity 0.801) sit at the null's 99th percentile.

Candidates must also have at least one real Party author, mirroring
``measure_wp_introduction.py``: without it an instrument frequently matches a
paper that merely reproduces its own adopted text.

Reading the output
------------------
Because candidates must predate the instrument, latency is >= 0 by construction.
Three things qualify every number:

* **chance** -- the latency from a randomly chosen *preceding* paper. The
  observed box should sit left of this; if it does not, the threshold has not
  bought any signal over picking arbitrarily.
* **ceiling** -- the latency from the oldest preceding paper, i.e. as far back as
  this instrument could possibly reach. A box riding its ceiling has degenerated
  into "find the oldest paper in the corpus".
* **unmatched** -- instruments with no preceding paper above the threshold are
  dropped, and they are not dropped at random. The bias is driven by instrument
  *type*, not by how much history was available: 80% of Recommendations and 71%
  of Measures match, against 52% of Decisions and 10% of untyped instruments.
  Over all types the unmatched share is not even monotonic in time (worst in the
  1990s-2000s); restricted to Measures it runs the other way, from 67% unmatched
  in the 1990s down to 10% in the 2020s. Every topic label carries
  ``matched/total`` and topics resting on under half their instruments are
  dimmed, so a latency computed from a minority of a topic cannot be misread as
  covering all of it.

Outputs (to ``data/latencies/``):
  * ``latency_matches.csv``     — one row per instrument: its match and latency
  * ``latency_by_topic.csv``    — per-topic n / mean / quartiles / chance / ceiling
  * ``latency_by_topic.png``    — box-and-whisker over all reported topics
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
from utils import line_buffer_stdout
from adhoc_analyses.measure_wp_topics import (
    fit_combined_topic_model,
    load_measures,
    load_working_papers,
)
from antarctic_ladder_metrics.measure_wp_introduction import NON_PARTY_AUTHORS

# Shared with latency_threshold_exploration.py, which imports it from here.
OUTPUT_DIR = pathlib.Path("data/latencies")

# Instrument types the latency is computed for; None means every type. Set it to
# a tuple such as ("Measure",) to narrow. Coverage differs sharply by type --
# 80% of Recommendations and 71% of Measures find a match, against 61% of
# Resolutions, 52% of Decisions and 10% of untyped instruments -- because
# hortatory instruments do not restate a working paper's text closely. That is
# the same poorly-matched tail the Annex-V investigation found, so read any
# all-type result as a blend of quite different matching regimes.
#
# The topic model is always fitted over every instrument plus every working
# paper, so topics stay identical to measure_wp_topics.py regardless of this
# setting; only the latency is filtered.
INSTRUMENT_TYPES = None

# A topic needs at least this many matched instruments before a box is drawn --
# a box plot over one or two points shows spread that isn't there.
MIN_MATCHES_FOR_BOX = 3

# Agreed cutoff for backward-looking matching (2026-07-21), from
# latency_threshold_exploration.py. Random instrument-paper pairs score a median
# cosine of 0.539 and a p99.9 of 0.846 under these embeddings, so 0.85 is
# stricter than 999 of 1000 pairs drawn at random. Below it the "earliest match
# above threshold" rule degenerates into "the oldest paper in the corpus": the
# median lag divided by how far back the instrument could reach runs 0.04 at
# 0.85, 0.19 at 0.80, 0.54 at 0.75, 0.89 at 0.70, so lags start growing with the
# calendar purely because later instruments have more history behind them.
# Coverage at 0.85 is 65% of instruments (71% of Type=="Measure"); note the
# dropped 35% skew early, since early instruments have less history to match.
SIMILARITY_THRESHOLD = 0.85

# Sequential single-hue palette for the observed distribution, plus one
# categorical partner for the chance reference. Validated (CVD dE 24.7).
HUE = "#2a78d6"
FILL = "#c8ddf5"
CHANCE = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d9d6"


def instrument_noun(plural: bool = False) -> str:
    """What to call the filtered instruments in titles and printed output."""
    if INSTRUMENT_TYPES is None:
        return "ATCM instruments" if plural else "ATCM instrument"
    joined = "/".join(INSTRUMENT_TYPES)
    return f"{joined}s" if plural else joined


def _has_real_party(working_paper: dict) -> bool:
    """True if any author is an actual Party rather than an observer body.

    ``parties`` arrives from the parquet as a numpy array, so it is normalised to
    a list before the usual splitting -- truthiness on an array is ambiguous.
    """
    parties = working_paper.get("parties")
    if parties is None:
        return False
    parties = [str(p) for p in list(parties)]
    return any(p not in NON_PARTY_AUTHORS for p in utils.split_parties(parties))


def eligible_wp_years(working_papers: list[dict]) -> np.ndarray:
    """Meeting years of the working papers the matcher is allowed to choose from."""
    years = np.array(
        [np.nan if pd.isna(w["year"]) else float(w["year"]) for w in working_papers]
    )
    ok = np.array([_has_real_party(w) for w in working_papers]) & ~np.isnan(years)
    return years[ok]


def label_order(labels) -> np.ndarray:
    """Column permutation that sorts candidates by label.

    Reorder a similarity matrix's columns by this before ``argmax`` and ties resolve by label
    instead of by array position -- ``argmax`` returns the *first* maximal index, so putting the
    columns in label order makes "first" mean "lexicographically smallest label".

    Position is not a defensible tie-break here. It is the order ``load_working_papers`` happened to
    build, which is ``get_embeddings_by_type``'s ``ORDER BY document_uuid`` -- so it is stable for a
    fixed store, but the uuid is a sha256 of the document's text. Re-OCR a paper, or change the
    segmentation, and its uuid moves, silently re-resolving ties between two *other* papers that
    did not change at all. Sorting by label pins the choice to something meaningful and stable
    across rebuilds.
    """
    return np.argsort(np.asarray(labels), kind="stable")


def argmax_tiebroken(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    """``values.argmax(axis=-1)`` with ties broken by label, via ``label_order``'s permutation.

    Works on a single row of similarities or a whole matrix; the returned indices are into the
    original, unpermuted columns.
    """
    return order[values[..., order].argmax(axis=-1)]


def match_instruments(measures: list[dict], working_papers: list[dict]) -> pd.DataFrame:
    """Earliest preceding working paper above the similarity threshold."""
    wp_years = np.array(
        [np.nan if pd.isna(w["year"]) else float(w["year"]) for w in working_papers]
    )
    wp_matrix = np.vstack([w["embedding"] for w in working_papers])
    wp_ok = np.array([_has_real_party(w) for w in working_papers]) & ~np.isnan(wp_years)
    wp_order = label_order([w["label"] for w in working_papers])

    rows = []
    for measure in measures:
        if pd.isna(measure["year"]):
            continue
        year = float(measure["year"])

        # Unit-norm vectors, so the dot product is the cosine similarity.
        sims = wp_matrix @ np.asarray(measure["embedding"], dtype="float32")
        preceding = wp_ok & (wp_years <= year)
        qualifying = preceding & (sims >= SIMILARITY_THRESHOLD)

        row = {
            "instrument": measure["label"],
            "instrument_type": measure["instrument_type"],
            "instrument_year": int(year),
            "adoption_year": measure["adoption_year"],
            "n_candidates": int(qualifying.sum()),
            # How far back this instrument could possibly have reached.
            "available_history": (
                int(year - wp_years[preceding].min()) if preceding.any() else None
            ),
            "matched": bool(qualifying.any()),
        }

        if qualifying.any():
            earliest_year = wp_years[qualifying].min()
            # Several papers can share the earliest year; take the most similar, and where two of
            # those are equally similar take the lexicographically first label rather than
            # whichever the store happened to return first (see label_order).
            tie = qualifying & (wp_years == earliest_year)
            best = int(argmax_tiebroken(np.where(tie, sims, -np.inf), wp_order))
            row.update(
                {
                    "matched_wp": working_papers[best]["label"],
                    "matched_wp_year": int(earliest_year),
                    "similarity": float(sims[best]),
                    "latency_years": int(year - earliest_year),
                }
            )
        else:
            row.update(
                {"matched_wp": None, "matched_wp_year": None,
                 "similarity": None, "latency_years": None}
            )
        rows.append(row)
    return pd.DataFrame(rows)


def chance_latency_by_topic(matches: pd.DataFrame, wp_years: np.ndarray) -> dict:
    """Median latency a topic would show if its papers were picked at random.

    The matcher may only look backwards, so the null draws from *preceding*
    papers only: the candidate latencies for one instrument are
    ``instrument_year - wp_years`` restricted to papers that predate it.

    Each instrument contributes its own median and the topic's figure is the
    median of those -- one vote per instrument, which is how ``median_latency``
    weights them too. Concatenating every instrument's candidates into a single
    pool instead weights each instrument by how many preceding papers it happens
    to have, and the corpus grows steeply over time (135 eligible papers precede
    1965, 2030 precede 2019), so a topic holding a 1965 and a 2019 instrument had
    93.8% of its pool contributed by the 2019 one. The bias has a direction: the
    instruments contributing the most rows are the late ones, which also have the
    deepest history behind them and so the largest chance lags. Pooling therefore
    pushed ``chance_latency`` up (18.0 rather than 11.0 on that pair) and
    ``median_vs_chance`` down, flattering the matcher.
    """
    chance = {}
    for topic, group in matches[matches["topic"] != -1].groupby("topic"):
        years = group.drop_duplicates("instrument")["instrument_year"].values
        per_instrument = [
            float(np.median(year - wp_years[wp_years <= year]))
            for year in years if (wp_years <= year).any()
        ]
        chance[topic] = float(np.median(per_instrument)) if per_instrument else np.nan
    return chance


def summarise_by_topic(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-topic latency stats, including how many instruments found no match.

    Grouping runs over *every* instrument, not just matched ones, so a topic
    where most instruments lacked a sufficiently close prior paper is visible
    rather than silently shrinking.
    """
    rows = []
    for topic, whole in all_rows[all_rows["topic"] != -1].groupby("topic"):
        group = whole[whole["matched"]]
        latencies = group["latency_years"]
        rows.append(
            {
                "topic": topic,
                "topic_label": whole["topic_label"].iloc[0],
                "n_total": len(whole),
                "n_unmatched": int((~whole["matched"]).sum()),
                "coverage": group.shape[0] / len(whole),
                "n_instruments": len(group),
                "mean_latency": latencies.mean(),
                "std_latency": latencies.std(ddof=0),
                "min_latency": latencies.min(),
                "q1_latency": latencies.quantile(0.25),
                "median_latency": latencies.median(),
                "q3_latency": latencies.quantile(0.75),
                "max_latency": latencies.max(),
                "ceiling_latency": group["available_history"].median(),
                "pct_at_ceiling": (group["latency_years"] == group["available_history"]).mean(),
                "mean_similarity": group["similarity"].mean(),
            }
        )
    # Many topics tie on median, so mean breaks the tie by tail length.
    return pd.DataFrame(rows).sort_values(["median_latency", "mean_latency"])


def plot_by_topic(matches: pd.DataFrame, summary: pd.DataFrame, chance: dict,
                  path: pathlib.Path) -> int:
    """Horizontal box plot, one row per topic, ordered by median latency.

    Horizontal because topic labels are long; ordered by median so the reader can
    scan fast-moving topics against slow ones without decoding position.
    """
    plotted = summary[summary["n_instruments"] >= MIN_MATCHES_FOR_BOX]
    if plotted.empty:
        return 0

    by_topic = {t: g["latency_years"].values for t, g in matches.groupby("topic")}
    data = [by_topic[t] for t in plotted["topic"]]
    # n matched / n instruments in the topic, so a topic whose latency rests on
    # a minority of its instruments cannot be read as if it covered them all.
    labels = [
        f"{row.topic_label}  ({row.n_instruments}/{row.n_total})"
        for row in plotted.itertuples()
    ]

    height = max(6.0, 0.30 * len(data) + 2.2)
    fig, ax = plt.subplots(figsize=(13, height))

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        widths=0.62,
        flierprops={"marker": "o", "markersize": 3.2, "markerfacecolor": HUE,
                    "markeredgecolor": "none", "alpha": 0.55},
    )
    for box in bp["boxes"]:
        box.set(facecolor=FILL, edgecolor=HUE, linewidth=1.2)
    for element in ("whiskers", "caps"):
        for item in bp[element]:
            item.set(color=HUE, linewidth=1.2)
    for median in bp["medians"]:
        median.set(color=INK, linewidth=2.0)

    positions = np.arange(1, len(data) + 1)
    ax.scatter([chance[t] for t in plotted["topic"]], positions,
               marker="D", s=15, facecolor="none", edgecolor=CHANCE,
               linewidths=1.3, zorder=5, label="chance (random preceding paper)")
    ax.scatter(plotted["ceiling_latency"], positions,
               marker="|", s=90, color=INK_MUTED, linewidths=1.4, zorder=4,
               label="ceiling (oldest preceding paper)")
    ax.plot([], [], color=HUE, linewidth=6, alpha=0.45,
            label=f"observed (earliest match ≥ {SIMILARITY_THRESHOLD})")
    # Directly above the axes. An in-axes legend collides with the ceiling ticks
    # (which sit far right), and anything offset below the axes moves with the
    # figure height, since these charts range from 22 rows to 90+.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.002), frameon=False,
              fontsize=9, labelcolor=INK_MUTED, handletextpad=0.8, ncol=3)

    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    # Second cue beyond the printed ratio: topics resting on under half their
    # instruments are dimmed, so weak rows do not read as strong ones.
    for tick, coverage in zip(ax.get_yticklabels(), plotted["coverage"]):
        if coverage < 0.5:
            tick.set_color("#8a8984")
            tick.set_style("italic")

    ax.set_xlabel("Years from earliest sufficiently-similar preceding working paper",
                  fontsize=10, color=INK_MUTED)
    latencies = matches.loc[matches["topic"] != -1, "latency_years"]
    ax.set_title(
        f"Working paper → {instrument_noun()} latency, by topic "
        f"(similarity ≥ {SIMILARITY_THRESHOLD})\n"
        f"{len(latencies)} matched · median {latencies.median():.0f}y · mean {latencies.mean():.1f}y · "
        f"labels show matched/total — italic grey = under half the topic's matched",
        fontsize=13, color=INK, pad=30, loc="left",
    )
    ax.set_xlim(left=-1)

    # Recessive grid on the value axis only; no competing horizontal rules.
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return len(data)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached embeddings...")
    measures = load_measures()
    working_papers = load_working_papers()
    docs = measures + working_papers
    print(f"  {len(measures)} instruments, {len(working_papers)} English working papers")

    topic_model, topics = fit_combined_topic_model(docs)
    topic_of = {id(d): t for d, t in zip(docs, topics)}

    def topic_label(topic: int) -> str:
        words = topic_model.get_topic(topic)
        if not isinstance(words, list):
            return str(topic)
        return ", ".join(w for w, _ in words[:4])

    selected = (
        measures if INSTRUMENT_TYPES is None
        else [m for m in measures if m["instrument_type"] in INSTRUMENT_TYPES]
    )
    print(f"  latency computed over {len(selected)} of {len(measures)} instruments "
          f"({instrument_noun(plural=True)})")

    all_rows = match_instruments(selected, working_papers)
    measure_topic = {m["label"]: topic_of[id(m)] for m in measures}
    all_rows["topic"] = all_rows["instrument"].map(measure_topic)
    all_rows["topic_label"] = all_rows["topic"].map(
        lambda t: "outliers" if t == -1 else topic_label(t)
    )
    all_rows.to_csv(OUTPUT_DIR / "latency_matches.csv", index=False)

    matches = all_rows[all_rows["matched"]].copy()
    summary = summarise_by_topic(all_rows)
    chance = chance_latency_by_topic(matches, eligible_wp_years(working_papers))
    summary["chance_latency"] = summary["topic"].map(chance)
    summary["median_vs_chance"] = summary["median_latency"] - summary["chance_latency"]
    summary.to_csv(OUTPUT_DIR / "latency_by_topic.csv", index=False)
    n_boxes = plot_by_topic(matches, summary, chance, OUTPUT_DIR / "latency_by_topic.png")

    print(f"\nmatched {len(matches)} of {len(all_rows)} instruments "
          f"({len(matches) / len(all_rows):.1%} coverage at similarity >= {SIMILARITY_THRESHOLD})")
    print(f"latency years  mean {matches.latency_years.mean():.1f}  "
          f"median {matches.latency_years.median():.0f}  "
          f"q3 {matches.latency_years.quantile(0.75):.0f}  max {matches.latency_years.max()}")
    print(f"matched-paper similarity  mean {matches.similarity.mean():.3f}  "
          f"min {matches.similarity.min():.3f}")

    unmatched = int((~all_rows["matched"]).sum())
    print(f"\nno sufficiently close prior working paper: {unmatched} of {len(all_rows)} "
          f"{instrument_noun(plural=True)} ({unmatched / len(all_rows):.1%})")
    if all_rows["instrument_type"].nunique() > 1:
        print("  by instrument type:")
        for t, g in all_rows.groupby("instrument_type"):
            n_un = int((~g["matched"]).sum())
            print(f"    {t:<16} {n_un:>4} of {len(g):<4} unmatched ({n_un / len(g):>5.1%})")
    print("  by instrument decade:")
    for d, g in all_rows.groupby(all_rows.instrument_year // 10 * 10):
        n_un = int((~g["matched"]).sum())
        print(f"    {d:<16} {n_un:>4} of {len(g):<4} unmatched ({n_un / len(g):>5.1%})")

    print(f"\ndegeneracy: {(matches.latency_years == matches.available_history).mean():.1%} "
          f"of matches reached the oldest paper available to them")

    print("\nmedian latency by instrument decade:")
    dec = matches.assign(decade=(matches.instrument_year // 10 * 10)).groupby("decade")
    print(dec.agg(n=("latency_years", "size"), median=("latency_years", "median"),
                  ceiling=("available_history", "median")).to_string())

    print(f"\n{len(summary)} topics with matches; {n_boxes} plotted "
          f"(>= {MIN_MATCHES_FOR_BOX} instruments)")
    print("\nTopics resting on the fewest of their instruments:")
    worst = summary[summary.n_instruments >= MIN_MATCHES_FOR_BOX].nsmallest(6, "coverage")
    print(worst[["topic", "n_instruments", "n_total", "n_unmatched", "coverage", "topic_label"]]
          .to_string(index=False, max_colwidth=44))

    cols = ["topic", "n_instruments", "n_total", "median_latency", "chance_latency",
            "ceiling_latency", "topic_label"]
    print("\nFastest topics by median latency:")
    print(summary[summary.n_instruments >= MIN_MATCHES_FOR_BOX].head(8)[cols]
          .to_string(index=False, max_colwidth=44))
    print("\nSlowest topics by median latency:")
    print(summary[summary.n_instruments >= MIN_MATCHES_FOR_BOX].tail(8)[cols]
          .to_string(index=False, max_colwidth=44))
    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    line_buffer_stdout()
    main()
