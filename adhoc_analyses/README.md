# Ad-hoc analyses

Exploratory one-off analyses that are not part of the antarctic ladder pipeline
(`antarctic_ladder_figure_aggregator.py`). Nothing here feeds `data/ladder_results.csv`.

Run from the repo root, e.g.:

```
uv run python -m adhoc_analyses.measure_wp_topics
```

Outputs land in `adhoc_analyses/output/`.

## Contents

- `measure_wp_topics.py` — BERTopic over the *combined* space of ATCM instruments
  ("measures", all four instrument types) and Working Papers, then the
  instrument/WP composition of each topic. Groundwork for measuring how long a
  topic takes to travel between the two document classes.

- `measure_wp_embedding_geometry.py` — cosine-distance geometry of the
  not-yet-effective measures vs Working Papers (WP-WP, measure-measure, WP-measure,
  and each measure's nearest WP). Written to probe whether the WP authorship
  classifiers fail on measures because the measures are OOD in the embedding space
  or merely lack the authorship signal. Finding: measures sit *inside* the WP
  manifold (every measure has a WP neighbour ~0.10 away, vs a ~0.43 typical WP-WP
  gap), so it is a decision-boundary problem, not OOD coverage.

The latency analyses that build on this (matching instruments to preceding
working papers, and the similarity-threshold exploration) now live in
`latency_analyses/`; they import the shared loaders from `measure_wp_topics.py`.
