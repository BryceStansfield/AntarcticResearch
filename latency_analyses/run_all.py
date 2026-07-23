"""Run every latency analysis in this folder, writing all outputs to data/latencies/.

The single entry point for the folder. Each analysis is self-contained -- it
loads the cached embeddings it needs and writes its own CSVs/figures -- so this
just runs them in sequence and reports which produced what.

    uv run python -m latency_analyses.run_all

The threshold exploration and the lag distributions both depend on
``SIMILARITY_THRESHOLD`` living in ``measure_wp_latency.py``, so that runs first;
otherwise the order is independent.
"""

from latency_analyses import (
    latency_threshold_exploration,
    lag_distributions,
    measure_wp_latency,
)

# (name, entry point) in run order.
ANALYSES = [
    ("Working paper → instrument latency, by topic", measure_wp_latency.main),
    ("Similarity-threshold exploration", latency_threshold_exploration.main),
    ("Lag distributions by decade and instrument type", lag_distributions.main),
]


def main():
    for i, (name, run) in enumerate(ANALYSES, start=1):
        banner = f"[{i}/{len(ANALYSES)}] {name}"
        print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}")
        run()
    print(f"\nAll {len(ANALYSES)} latency analyses complete.")


if __name__ == "__main__":
    main()
