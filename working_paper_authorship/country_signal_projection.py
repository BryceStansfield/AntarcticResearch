"""Project embeddings into the subspace orthogonal to the direct country-authorship signal.

``direct_country_signal_probe`` recovered, by injection, a stable per-country direction that an
explicit authorship statement moves the embedding along (see its report / the saved
``direct_country_directions_allwps.npz``). Those 5 directions span a low-dimensional "direct country
signal" subspace S of the 4096-dim embedding space. This class builds an orthonormal basis of S and
projects it out: P x = x - B Bᵀ x, leaving the component of every embedding orthogonal to the direct
signal — the ambient space is unchanged in dimension, but the directions that explicitly encode
"authored by <country>" are nulled.

Note the country directions are not mutually orthogonal (UK and US in particular share an axis,
cosine ~0.36), so we orthonormalise the *span* via SVD rather than treating each direction
independently — projecting out the subspace handles the overlap correctly and removes exactly one
shared axis for the UK/US pair instead of double-counting it.

The span is (n_countries - 1)-dimensional, not n_countries: the probe centres each document's
per-country shift across countries, so the directions sum to zero. See ``DEFAULT_TOL``.

``project`` re-normalises, so orthogonalised embeddings stay unit-norm like the ones they came
from. Consumers treat a dot product as a cosine (``measure_wp_latency`` says so and thresholds on
it), and the length a projection removes varies per document, so leaving it out would make
documents with more country signal look less similar to everything for purely geometric reasons.
"""
import pathlib

import numpy as np


class CountrySignalProjector:
    """Orthogonal projector that removes the direct country-signal subspace from embeddings."""

    # Relative singular-value floor for deciding which directions are real.
    #
    # The country directions are rank-deficient *by construction*, not by accident. The probe builds
    # them as ``country = delta - delta.mean(axis=1)`` -- centred across countries -- so the C
    # vectors sum to zero and span exactly C-1 dimensions. Normalising each to unit length
    # afterwards does not change that: scaling individual vectors never changes their span.
    #
    # So the C-th singular value should be zero, and what is actually there is float32 round-off:
    # the probe accumulates in float32 (eps 1.19e-7), and the residual measures ~5e-7 relative.
    # The previous 1e-8 was a sensible float64 threshold applied to float32-derived data, so it
    # kept that noise and reported rank C where the subspace is C-1 -- projecting out one extra,
    # essentially arbitrary direction and misreporting the rank downstream.
    #
    # The value is not delicate. Real axes sit at s/s[0] ~ 0.97 and the residual at ~5e-7, six
    # orders of magnitude apart, so anything from ~1e-6 to ~1e-2 selects the same basis.
    DEFAULT_TOL = 1e-5

    # What counts as "nothing left after the projection", as a fraction of the row's original
    # length. Sits above float32's ~1.2e-7 round-off and far below any real document's residual,
    # which is essentially 1.0 -- a 4-dimensional subspace removed from 4096 dimensions.
    RESIDUAL_TOL = 1e-6

    def __init__(self, directions: np.ndarray, tol: float = DEFAULT_TOL):
        """directions: (n_countries, dim) — the per-country mean direct-signal directions (any scale)."""
        self.basis = self._orthonormal_basis(np.asarray(directions, dtype=np.float64), tol)  # (rank, dim)
        self.rank = int(self.basis.shape[0])
        self.dim = int(self.basis.shape[1])

    @staticmethod
    def _orthonormal_basis(directions: np.ndarray, tol: float) -> np.ndarray:
        """Orthonormal basis of the row-span of ``directions`` (drops numerically-dependent rows)."""
        # Right singular vectors whose singular value is non-negligible span the row space.
        _u, s, vt = np.linalg.svd(directions, full_matrices=False)
        keep = s > (tol * s[0]) if s.size else np.array([], dtype=bool)
        return vt[keep]

    def project(self, X: np.ndarray) -> np.ndarray:
        """The component of each row of X orthogonal to the country subspace, re-normalised.

        Removing a component genuinely shortens a vector, and by an amount that varies per document
        -- a paper whose content lies largely along the country directions loses much more length
        than one that barely touches them. Left unnormalised that becomes a magnitude artefact
        masquerading as a semantic one: every downstream dot product against such a document is
        systematically smaller, so it looks less similar to everything, purely because it had more
        country signal to remove.

        That matters because the consumers assume unit norm. ``measure_wp_latency`` states it
        outright -- "Unit-norm vectors, so the dot product is the cosine similarity" -- and then
        thresholds on that dot product at 0.85, so shortened vectors would match less often for a
        reason that has nothing to do with content. Source embeddings arrive unit-norm, so
        re-normalising here keeps that invariant true through the projection, exactly as
        ``mean_pool`` does for segment pooling.

        A row lying inside the subspace has nothing left to normalise and comes back as the zero
        vector. The test for that is a *relative* tolerance, not ``norm > 0``: in float32 the
        projection of such a row leaves round-off of order 1e-7 rather than an exact zero, and
        dividing by that would rescale pure numerical noise into a full-length unit vector pointing
        in an arbitrary direction -- worse than the shortened vector normalising is meant to fix.
        Real documents are nowhere near this: removing a 4-dimensional subspace from a 4096-
        dimensional embedding leaves essentially all of its length, so the guard only ever fires on
        degenerate input.
        """
        X = np.asarray(X, dtype=np.float32)
        B = self.basis.astype(np.float32)
        projected = X - (X @ B.T) @ B

        norms = np.linalg.norm(projected, axis=-1, keepdims=True)
        source_norms = np.linalg.norm(X, axis=-1, keepdims=True)
        meaningful = norms > self.RESIDUAL_TOL * source_norms
        return np.divide(projected, norms, out=np.zeros_like(projected), where=meaningful)

    # Sklearn-style aliases so this can also drop into a Pipeline as a transformer if wanted.
    def fit(self, X=None, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.project(X)

    @classmethod
    def from_npz(cls, path: str | pathlib.Path, **kwargs) -> "CountrySignalProjector":
        """Load the country directions saved by ``direct_country_signal_probe`` and build the projector."""
        data = np.load(pathlib.Path(path), allow_pickle=True)
        return cls(data["directions"], **kwargs)
