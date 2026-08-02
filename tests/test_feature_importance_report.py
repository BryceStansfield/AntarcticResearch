"""Tests for how the feature report ranks PCA components.

The report exists to test one hypothesis: that the authorship classifiers key on fine, non-topic
directions rather than on topic. So the ranking it prints has to be an honest measure of influence,
because the failure mode is not noise -- it is a bias pointing straight at the hypothesis.

A coefficient alone is not an influence when the features are not on a common scale. PCA components
are uncorrelated but deliberately un-standardised: their variances are the eigenvalues and here they
span orders of magnitude. What a component contributes to the logit is ``|coef| * sd``, so ranking on
``|coef|`` alone over-weights the low-variance tail -- exactly the "fine directions" the report is
supposed to be testing for rather than assuming.

`component_importances` is the only piece of that module worth pinning: the rest reads pickles off
disk and formats text. It is exercised against hand-built estimators (never fitted -- the values
under test are set directly) so the arithmetic is checked against numbers a reader can verify.
"""
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from working_paper_authorship.feature_importance_report import component_importances


def _pipeline(clf, explained_variance):
    """A PCA -> classifier pipeline with the PCA's component variances planted directly.

    `component_importances` reads `pca.explained_variance_` and the classifier's coefficients and
    nothing else, so fitting real data would only obscure which numbers drive the result.
    """
    pca = PCA()
    pca.explained_variance_ = np.asarray(explained_variance, dtype=float)
    return Pipeline([("pca", pca), ("clf", clf)])


def _linear_clf(coefs_per_label):
    """A `MultiOutputClassifier` whose per-label estimators carry the given coefficients."""
    clf = MultiOutputClassifier(LogisticRegression())
    estimators = []
    for coefs in coefs_per_label:
        estimator = LogisticRegression()
        estimator.coef_ = np.asarray([coefs], dtype=float)
        estimators.append(estimator)
    clf.estimators_ = estimators
    return clf


def test_coefficients_are_scaled_by_component_standard_deviation():
    """Influence on the logit is |coef| * sd, and sd is sqrt of the component's variance."""
    clf = _linear_clf([[2.0, 1.0]])
    pipeline = _pipeline(clf, explained_variance=[4.0, 100.0])

    importances, method = component_importances(pipeline, X_val=None, Y_val=None)

    # component 0: |2| * sqrt(4) = 4     component 1: |1| * sqrt(100) = 10
    assert importances == pytest.approx([4.0, 10.0])
    assert "sd" in method


def test_a_high_variance_component_can_outrank_a_larger_coefficient():
    """The regression, in miniature. Ranking on |coef| alone puts component 0 first; once each is
    weighted by how much it actually varies, the high-variance component 1 is the real contributor.
    On logistic_regression__raw__full this was the difference between reporting components
    [16, 13, 28, 30, 22] and the true top contributors [1, 0, 6, 16, 13] -- the two highest-variance,
    most plausibly topical components did not appear at all."""
    clf = _linear_clf([[10.0, 1.0]])
    pipeline = _pipeline(clf, explained_variance=[0.01, 400.0])

    importances, _ = component_importances(pipeline, X_val=None, Y_val=None)

    assert np.argmax(importances) == 1
    assert np.argmax([10.0, 1.0]) == 0, "unscaled, the ranking is the other way round"


def test_coefficients_are_averaged_across_country_labels():
    """One estimator per country; a component's importance is its mean |coef| over them, so a
    direction that only one country's model leans on is not ranked as if every model did."""
    clf = _linear_clf([[4.0, 0.0], [0.0, 0.0]])
    pipeline = _pipeline(clf, explained_variance=[1.0, 1.0])

    importances, _ = component_importances(pipeline, X_val=None, Y_val=None)

    assert importances == pytest.approx([2.0, 0.0])


def test_signs_do_not_cancel_across_labels():
    """Two countries leaning oppositely on one component both use it; averaging the raw coefficients
    would report it as unused."""
    clf = _linear_clf([[3.0, 0.0], [-3.0, 0.0]])
    pipeline = _pipeline(clf, explained_variance=[1.0, 1.0])

    importances, _ = component_importances(pipeline, X_val=None, Y_val=None)

    assert importances[0] == pytest.approx(3.0)


def test_tree_importances_are_taken_as_they_are():
    """Impurity/gain importances are already computed over the split values, so they are scale-aware
    and must not be scaled a second time."""
    class _Forest:
        feature_importances_ = np.asarray([0.25, 0.75])

    pipeline = _pipeline(_Forest(), explained_variance=[0.01, 400.0])

    importances, method = component_importances(pipeline, X_val=None, Y_val=None)

    assert importances == pytest.approx([0.25, 0.75])
    assert "native" in method
