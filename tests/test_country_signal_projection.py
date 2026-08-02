"""Tests for the direct country-signal projector.

The subspace it removes is rank-deficient by construction. `direct_country_signal_probe` builds
each document's per-country vectors as ``country = delta - delta.mean(axis=1)`` — centred across
countries — so the C directions sum to zero and span exactly C-1 dimensions. Per-country
normalisation afterwards does not change that, since scaling individual vectors never changes their
span.

So the C-th singular value should be zero, and what is actually present is float32 round-off: the
probe accumulates in float32 (eps 1.19e-7) and the residual measures ~5e-7 relative to the largest.
The old 1e-8 tolerance was a float64 threshold applied to float32-derived data, so it kept that
noise, projected out one extra arbitrary direction, and reported rank C where the subspace is C-1.

These tests rebuild the probe's pipeline rather than reading a saved npz, so they pin the
*structure* — they hold whatever the current directions file happens to contain.
"""
import numpy as np
import pytest

from working_paper_authorship.country_signal_projection import CountrySignalProjector

C, DIM, N_DOCS = 5, 256, 400


def probe_style_directions(seed=0, dim=DIM, n_countries=C):
    """Directions built exactly the way the probe builds them, in float32.

    Centre each document's per-country shift across countries, average over documents, then
    normalise each country's mean to unit length.
    """
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=(N_DOCS, n_countries, dim)).astype(np.float32)
    country = delta - delta.mean(axis=1, keepdims=True)
    dirs = []
    for c in range(n_countries):
        m = country[:, c, :].mean(axis=0)
        dirs.append(m / np.linalg.norm(m))
    return np.array(dirs, dtype=np.float32)


# --------------------------------------------------------------------------- rank selection

def test_centred_directions_span_one_fewer_dimension_than_countries():
    """The structural property: C centred directions span C-1 dimensions."""
    projector = CountrySignalProjector(probe_style_directions())
    assert projector.rank == C - 1


def test_the_old_tolerance_kept_the_float32_residual():
    """Contrast, so the test above is not vacuous: 1e-8 admits the round-off as a real axis."""
    directions = probe_style_directions()
    assert CountrySignalProjector(directions, tol=1e-8).rank == C
    assert CountrySignalProjector(directions, tol=CountrySignalProjector.DEFAULT_TOL).rank == C - 1


def test_lowering_the_tolerance_keeps_more_not_fewer():
    """The tolerance is a floor on s/s[0], so a smaller value is a lower bar. Pinned because it
    reads backwards: 'tighten the tolerance' means raise this number, not lower it."""
    directions = probe_style_directions()
    ranks = [CountrySignalProjector(directions, tol=t).rank for t in (1e-10, 1e-8, 1e-6, 1e-4)]
    assert ranks == sorted(ranks, reverse=True)
    assert ranks[0] == C and ranks[-1] == C - 1


def test_the_choice_of_tolerance_is_not_delicate():
    """Real axes and the residual are ~6 orders of magnitude apart, so any threshold across that
    gap selects the same basis. If this ever fails, the directions have changed shape."""
    directions = probe_style_directions()
    assert {CountrySignalProjector(directions, tol=t).rank
            for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)} == {C - 1}


def test_genuinely_independent_directions_are_all_kept():
    """The tolerance must not over-prune: uncentred, independent directions are full rank."""
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(C, DIM)).astype(np.float32)
    assert CountrySignalProjector(directions).rank == C


def test_an_exactly_duplicated_direction_is_dropped():
    rng = np.random.default_rng(2)
    directions = rng.normal(size=(C, DIM))
    directions[1] = directions[0]
    assert CountrySignalProjector(directions).rank == C - 1


# --------------------------------------------------------------------------- the projection

def test_the_basis_is_orthonormal():
    basis = CountrySignalProjector(probe_style_directions()).basis
    assert basis @ basis.T == pytest.approx(np.eye(basis.shape[0]), abs=1e-9)


def test_projected_vectors_are_orthogonal_to_every_country_direction():
    """What the projector is for: no trace of any country direction survives in the output."""
    directions = probe_style_directions()
    projector = CountrySignalProjector(directions)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, DIM)).astype(np.float32)

    projected = projector.project(X)
    assert projected @ directions.T == pytest.approx(np.zeros((20, C)), abs=2e-5)


def test_projection_is_idempotent():
    projector = CountrySignalProjector(probe_style_directions())
    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, DIM)).astype(np.float32)

    once = projector.project(X)
    assert projector.project(once) == pytest.approx(once, abs=1e-5)


def test_a_vector_already_orthogonal_to_the_subspace_is_untouched():
    directions = probe_style_directions()
    projector = CountrySignalProjector(directions)
    rng = np.random.default_rng(5)

    x = rng.normal(size=DIM).astype(np.float32)
    orthogonal = projector.project(x[None, :])
    assert projector.project(orthogonal) == pytest.approx(orthogonal, abs=1e-5)


def test_a_direction_in_the_subspace_projects_to_nothing():
    directions = probe_style_directions()
    projector = CountrySignalProjector(directions)
    projected = projector.project(directions[:1].astype(np.float32))
    assert np.linalg.norm(projected) == pytest.approx(0.0, abs=2e-5)


def test_project_does_not_renormalise():
    """Characterises current behaviour, and it is a live decision rather than settled: removing a
    component genuinely shortens the vector, so projected embeddings are NOT unit-norm. Consumers
    that read a dot product as a cosine (measure_wp_latency) must normalise themselves."""
    directions = probe_style_directions()
    projector = CountrySignalProjector(directions)
    rng = np.random.default_rng(6)

    x = rng.normal(size=(1, DIM)).astype(np.float32)
    x = x / np.linalg.norm(x)
    assert np.linalg.norm(projector.project(x)) < 1.0


# --------------------------------------------------------------------------- loading

def test_from_npz_round_trips(tmp_path):
    directions = probe_style_directions()
    path = tmp_path / "directions.npz"
    np.savez(path, directions=directions)

    assert CountrySignalProjector.from_npz(path).rank == C - 1
    assert CountrySignalProjector.from_npz(path, tol=1e-8).rank == C
