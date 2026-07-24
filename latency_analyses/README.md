# Latency analyses

Working-Paper → ATCM-instrument latency: how long an instrument took to arrive
after the working paper that anticipated it. These are exploratory analyses, not
part of the antarctic ladder pipeline (`antarctic_ladder_figure_aggregator.py`);
nothing here feeds `data/ladder_results.csv`.

The shared loaders (`load_measures`, `load_working_papers`,
`fit_combined_topic_model`) live in `adhoc_analyses/measure_wp_topics.py` and are
imported from there, so the topic model stays identical to that analysis.

Run the whole folder from the repo root:

```
uv run python -m latency_analyses.run_all
```

or any single analysis, e.g. `uv run python -m latency_analyses.measure_wp_latency`.
Outputs land in `data/latencies/`.

## Contents

- `run_all.py` — runs every analysis below in sequence.
- `measure_wp_latency.py` — matches every instrument to the earliest preceding
  working paper scoring at least `SIMILARITY_THRESHOLD`, then summarises the
  resulting latencies per topic.
- `latency_threshold_exploration.py` — sweeps the similarity threshold to check
  that backward-looking matching does not degenerate into "the oldest paper in
  the corpus". This is where `SIMILARITY_THRESHOLD = 0.85` was chosen. Also emits
  `threshold_decade_family.png`, a full-size version of its by-decade panel with
  one line per threshold from 0.75 to 0.90.
- `lag_distributions.py` — box-and-whisker plots of the matched lags, two views:
  `lag_box_by_decade.png` (one panel per decade, a box per instrument type) and
  `lag_box_by_type.png` (one panel per instrument type, a box per decade).
