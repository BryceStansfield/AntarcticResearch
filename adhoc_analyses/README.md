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
