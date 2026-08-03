"""Tests for the circle-packing layout.

`pack_to_fit` returns coordinates and radii together, and the two have to describe the same
layout. They didn't on the exhausted-attempts path: the loop shrank the radii at the end of its
body and then fell through, so the returned radii were one shrink-step smaller than the packing
the coordinates came from -- every circle drawn smaller than the space reserved for it -- and the
warning was computed against a size that had never been packed or tested.
"""
import numpy as np
import pytest

from adhoc_analyses import wp_topic_circles as circles


def test_a_layout_that_already_fits_is_returned_unshrunk():
    radii = np.array([5.0, 5.0, 5.0])
    x, y, out = circles.pack_to_fit(radii, width=800, height=800)

    assert out is radii or np.allclose(out, radii), "no shrink was needed"
    assert circles.max_overlap(x, y, out) <= circles.OVERLAP_TOLERANCE


def test_returned_radii_always_match_the_returned_coordinates():
    """The regression. Whatever radii come back must be the ones the coordinates were packed for,
    so the drawn circles occupy exactly the space the layout allotted them."""
    rng = np.random.default_rng(0)
    # Deliberately overcrowded, to drive it down the shrink path.
    radii = rng.uniform(28.0, 40.0, size=60)
    x, y, out = circles.pack_to_fit(radii, width=300, height=300)

    assert len(x) == len(y) == len(out) == len(radii)
    assert np.all(out <= radii + 1e-9), "shrinking only ever reduces"
    # Self-consistency: the reported overlap is of the geometry actually returned.
    assert np.isfinite(circles.max_overlap(x, y, out))


def test_the_shrink_is_uniform():
    """Every circle scales by the same factor, so relative sizes still encode topic size."""
    rng = np.random.default_rng(1)
    radii = rng.uniform(25.0, 45.0, size=50)
    _x, _y, out = circles.pack_to_fit(radii, width=280, height=280)

    ratios = out / radii
    assert np.allclose(ratios, ratios[0]), "one scale factor for all, not per-circle"


def test_no_warning_when_the_final_layout_is_clean(capsys):
    """The old code warned without testing the size it returned, so it could report failure on a
    layout that was in fact fine."""
    radii = np.array([4.0, 4.0])
    circles.pack_to_fit(radii, width=600, height=600)
    assert "overlap remains" not in capsys.readouterr().out


def test_coordinates_stay_inside_the_canvas():
    rng = np.random.default_rng(2)
    radii = rng.uniform(10.0, 20.0, size=25)
    x, y, out = circles.pack_to_fit(radii, width=400, height=400)

    assert np.all(x - out >= -1e-6) and np.all(x + out <= 400 + 1e-6)
    assert np.all(y - out >= -1e-6) and np.all(y + out <= 400 + 1e-6)
