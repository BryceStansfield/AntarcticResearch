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
"""
import pathlib

import numpy as np


class CountrySignalProjector:
    """Orthogonal projector that removes the direct country-signal subspace from embeddings."""

    def __init__(self, directions: np.ndarray, tol: float = 1e-8):
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
        """Return the component of each row of X orthogonal to the country subspace: X - (X Bᵀ) B."""
        X = np.asarray(X, dtype=np.float32)
        B = self.basis.astype(np.float32)
        return X - (X @ B.T) @ B

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
