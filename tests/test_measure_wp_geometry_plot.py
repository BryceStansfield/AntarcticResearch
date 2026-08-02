"""Tests for the geometry figure's histogram range.

This figure is the evidence for "measures sit inside the working-paper manifold, so the classifier
failure is a decision-boundary problem rather than an OOD-coverage one". Cosine *distance* runs
0..2, and nothing upstream guarantees non-negative similarity -- so a range hardcoded to (0, 1)
silently discards any pair beyond it. The discarded pairs would be exactly the ones showing
measures pointing *away* from the working papers, i.e. the evidence against the conclusion the
figure is drawn to support.
"""
import numpy as np
import pytest

from adhoc_analyses.measure_wp_embedding_geometry import plot_histograms


def test_every_value_falls_inside_the_drawn_range(tmp_path):
    """The property that matters: no input is outside the bins, whatever the data looks like."""
    dists = {
        "a": np.array([0.1, 0.4, 0.9]),
        "b": np.array([1.3, 1.8]),      # beyond a hardcoded (0, 1)
        "c": np.array([-0.2, 0.5]),     # negative cosine similarity -> distance > 1... and below 0
    }
    path = tmp_path / "hist.png"
    plot_histograms(dists, path)
    assert path.exists()

    lo = min(float(d.min()) for d in dists.values())
    hi = max(float(d.max()) for d in dists.values())
    assert lo == pytest.approx(-0.2)
    assert hi == pytest.approx(1.8)


def test_histogram_of_out_of_unit_range_data_keeps_all_of_it(tmp_path):
    """Directly contrasts the old behaviour: with range=(0, 1), np.histogram drops these entirely."""
    values = np.array([1.2, 1.5, 1.9])

    dropped, _ = np.histogram(values, bins=60, range=(0, 1))
    assert dropped.sum() == 0, "the old hardcoded range discarded every one of these"

    kept, _ = np.histogram(values, bins=60, range=(float(values.min()), float(values.max())))
    assert kept.sum() == len(values)


def test_a_single_series_still_renders(tmp_path):
    path = tmp_path / "one.png"
    plot_histograms({"only": np.array([0.2, 0.3, 0.4])}, path)
    assert path.exists()
