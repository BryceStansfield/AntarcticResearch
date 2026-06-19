"""A differentiable, multilabel RBF SVM built on Random Fourier Features (RFF).

The RBF kernel is approximated with random features phi(z) = sqrt(1/M) [cos(zW), sin(zW)]
(Rahimi & Recht), so the whole decision function is smooth in the input and exposes
gradients via PyTorch autograd. This lets downstream code walk the (unit-sphere)
embedding space toward a point that maximises the joint authorship probability
p_1 * ... * p_n, while still behaving like an sklearn classifier (fit / predict /
predict_proba) so it slots into the existing PCA -> clf pipeline, Optuna search and
result caching. It is natively multilabel: one model emits all party logits at once.

The torch-native, differentiable entry points operate in PCA space (the classifier's
input). Because PCA is itself linear, a full raw-embedding -> probability map can be
obtained by composing pca.mean_ / pca.components_ with predict_proba_torch.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


class RFFSVM(BaseEstimator, ClassifierMixin):
    """Multilabel RBF SVM via Random Fourier Features, trained with squared hinge loss.

    Parameters mirror an rbf SVC: ``C`` (regularisation strength) and ``gamma`` (kernel
    bandwidth). ``n_features`` is the number of random frequencies sampled; the cos/sin
    map yields ``2 * n_features`` actual features. Higher means a closer kernel
    approximation. Probabilities come from per-label Platt scaling fitted on the trained
    margins."""

    def __init__(self, n_features: int = 512, gamma: float = 0.1, C: float = 1.0,
                 lr: float = 0.05, epochs: int = 300, random_state: int = 0):
        self.n_features = n_features
        self.gamma = gamma
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

    def _random_features(self, Z: torch.Tensor) -> torch.Tensor:
        # Concatenated cos/sin map: phi(z) = sqrt(1/M) [cos(zW), sin(zW)], M = n_features.
        # For a given number of features this has lower approximation variance than the
        # cos(zW + b) offset construction — Sutherland & Schneider, "On the Error of
        # Random Fourier Features" (UAI 2015).
        projection = Z @ self.rff_weight_.to(Z)
        features = torch.cat([projection.cos(), projection.sin()], dim=-1)
        return features * (1.0 / self.n_features) ** 0.5

    def fit(self, X, Y):
        torch.set_num_threads(1)  # parallelism comes from the outer CV search
        generator = torch.Generator().manual_seed(self.random_state)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
        n_dims, self.n_labels_ = X.shape[1], Y.shape[1]
        signs = 2.0 * Y - 1.0  # {0,1} -> {-1,+1}

        # Fixed random frequencies for an RBF kernel of bandwidth gamma.
        self.rff_weight_ = torch.randn(n_dims, self.n_features, generator=generator) * (2.0 * self.gamma) ** 0.5
        features = self._random_features(X)

        # Trainable linear SVM head (one row per party), sized to the 2*M feature map.
        weight = torch.zeros(self.n_labels_, features.shape[1], requires_grad=True)
        bias = torch.zeros(self.n_labels_, requires_grad=True)
        optimiser = torch.optim.Adam([weight, bias], lr=self.lr)
        for _ in range(self.epochs):
            optimiser.zero_grad()
            decision = features @ weight.t() + bias
            squared_hinge = torch.clamp(1.0 - signs * decision, min=0.0) ** 2
            loss = 0.5 * (weight ** 2).sum() + self.C * squared_hinge.mean()
            loss.backward()
            optimiser.step()
        self.weight_ = weight.detach()
        self.bias_ = bias.detach()

        # Per-label Platt scaling: map margins to calibrated probabilities.
        with torch.no_grad():
            decision = features @ self.weight_.t() + self.bias_
        platt_a = torch.ones(self.n_labels_, requires_grad=True)
        platt_b = torch.zeros(self.n_labels_, requires_grad=True)
        optimiser = torch.optim.Adam([platt_a, platt_b], lr=0.05)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(200):
            optimiser.zero_grad()
            bce(platt_a * decision + platt_b, Y).backward()
            optimiser.step()
        self.platt_a_ = platt_a.detach()
        self.platt_b_ = platt_b.detach()

        self.classes_ = [np.array([0, 1]) for _ in range(self.n_labels_)]
        return self

    def decision_function_torch(self, Z: torch.Tensor) -> torch.Tensor:
        """Differentiable per-label margins for PCA-space inputs Z (n_samples, n_dims)."""
        return self._random_features(Z) @ self.weight_.to(Z).t() + self.bias_.to(Z)

    def predict_proba_torch(self, Z: torch.Tensor) -> torch.Tensor:
        """Differentiable per-label P(label=1), shape (n_samples, n_labels)."""
        decision = self.decision_function_torch(Z)
        return torch.sigmoid(self.platt_a_.to(Z) * decision + self.platt_b_.to(Z))

    def predict_proba(self, X) -> np.ndarray:
        Z = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            return self.predict_proba_torch(Z).numpy()

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    # Self-test: sklearn compatibility (clone / set_params / pickle in a search) and that
    # the joint-probability objective is differentiable w.r.t. the input.
    import pickle
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 24)).astype(np.float32)
    Y = (rng.random((120, 5)) < 0.4).astype(int)

    def subset_accuracy(estimator, Xv, Yv):
        return float((estimator.predict(Xv) == Yv).all(axis=1).mean())

    search = GridSearchCV(
        Pipeline([("pca", PCA(random_state=0)), ("clf", RFFSVM(random_state=0))]),
        param_grid={"pca__n_components": [4, 8], "clf__C": [0.1, 1.0]},
        scoring=subset_accuracy, cv=3, n_jobs=1,
    ).fit(X, Y)
    print("GridSearch best params:", search.best_params_)

    best = pickle.loads(pickle.dumps(search.best_estimator_))  # cache round-trip
    proba = best.predict_proba(X)
    print("predict_proba shape:", proba.shape, "range:", (round(float(proba.min()), 3), round(float(proba.max()), 3)))

    clf = best.named_steps["clf"]
    Z = torch.tensor(best.named_steps["pca"].transform(X[:4]), dtype=torch.float32, requires_grad=True)
    joint = clf.predict_proba_torch(Z).prod(dim=1).sum()  # p1 * ... * pn, summed over docs
    joint.backward()
    assert Z.grad is not None and torch.isfinite(Z.grad).all()
    print(f"joint prob objective={joint.item():.4f}, grad norm={Z.grad.norm().item():.4f}")
    print("SELF-TEST OK")
