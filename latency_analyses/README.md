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
- `rerank_latency_comparison.py` — a second opinion on the cosine ordering. Takes
  every dated instrument (set `N_INSTRUMENTS` to an int to sample instead), takes
  each one's 10 nearest working papers in embedding space (no precedence filter,
  so lags may be negative), and reorders that same set with a
  `cohere/rerank-4-pro` cross-encoder. Emits
  `rerank_latency_by_rank.png`, the lag distribution at each rank under cosine
  vs reranked order, and `rerank_latency_comparison.csv`.
- `wp_reranker.py` — the cached OpenRouter reranker used above. Scores are cached
  per (model, query, document) in `data/latencies/rerank_cache.sqlite3`, so only
  unseen instrument–paper pairs touch the network. Not an analysis; imported, not
  run.
