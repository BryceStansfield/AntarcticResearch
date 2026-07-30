# Tests

Pure-helper tests for the measure pipeline and the shared country/text utilities.
Run from the repository root:

    python -m pytest tests/ -q

Nothing here touches the network, the embedding caches or the real corpus:
`enrich_measure_data` is driven over temp-file fixtures, and everything else is a
pure function.

Tests labelled **characterises current behaviour** pin down a known wart rather
than assert something desirable. Each names the wart and why it is currently
harmless, so if you fix the underlying issue these are the tests expected to fail.

## What is not covered, and why

`RatificationSpeed`'s approval parsing — the country/date pairing loop and
`extract_end_year` — used to be the highest-risk untested logic in the measure
code: both lived inside `__init__`, which calls `scrape_and_enrich_measures`
and then reads the full corpus, so exercising the parser meant running the
whole pipeline. They have since been lifted to module-level functions
(`extract_end_year`, `parse_approval_pairs`, `compute_row_delays`) that take
`(approvals_text, atcm_year, status)` and have no dependency on the class, the
corpus, or the network, and are covered directly in
`tests/test_ratification_speed.py`.

`RatificationSpeed.__init__` itself, and the `add_approval` closure inside it
(the " *" Consultative Party population filter — see the comment above it in
`ratification_speed.py`), are still not covered: instantiating the class still
means running the real pipeline.
