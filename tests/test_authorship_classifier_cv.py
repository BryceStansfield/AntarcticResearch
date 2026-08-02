"""Tests for the two pieces of the authorship benchmark that decide what "best" means.

Both failed quietly rather than loudly, which is why they are worth pinning: the hyperparameter
search still ran, still reported a winner, and the winner was chosen against a corrupted score.

* `positive_proba` normalises the various `predict_proba` conventions into P(label==1). When a CV
  fold contains only one class for a rare label it has to work out *which* class the single
  returned column refers to -- and reading that from the wrong object made it always assume the
  negative one.
* `inner_cv` groups the CV by source document. A paper too long for the embedder's context window
  occupies several rows sharing one label, so an ungrouped split puts near-duplicate text on both
  sides of the fold.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputClassifier

from working_paper_authorship import country_authorship_classifier as cc


# --------------------------------------------------------------------------- positive_proba

class _SingleClassEstimator:
    """A per-label estimator that saw exactly one class, as happens for a rare label in a fold."""

    def __init__(self, only_class):
        self.classes_ = np.array([only_class])


class _Wrapper(MultiOutputClassifier):
    """A MultiOutputClassifier whose predict_proba is planted directly.

    Subclassed rather than mocked so `isinstance` and attribute lookup behave exactly as they do in
    the real pipeline -- the defect under test was precisely an attribute being looked up on this
    object instead of on its per-label estimators.
    """

    def __init__(self, proba, estimators):
        super().__init__(LogisticRegression())
        self._proba = proba
        self.estimators_ = estimators

    def predict_proba(self, X):
        return self._proba


def test_two_class_columns_take_the_positive_one():
    proba = [np.array([[0.3, 0.7], [0.8, 0.2]])]
    out = cc.positive_proba(_Wrapper(proba, [LogisticRegression()]), X=None)
    assert out.ravel() == pytest.approx([0.7, 0.2])


def test_an_all_positive_fold_scores_one_not_zero():
    """The regression. `MultiOutputClassifier` exposes no `classes_` of its own, so the old
    `getattr(estimator, "classes_", None)` was always None and this branch always read the single
    column as the negative class -- scoring a fold that was entirely positive as P(author)=0.0 for
    every row, i.e. maximally wrong, and feeding that to the hyperparameter search."""
    proba = [np.array([[1.0], [1.0]])]
    out = cc.positive_proba(_Wrapper(proba, [_SingleClassEstimator(1)]), X=None)
    assert out.ravel() == pytest.approx([1.0, 1.0])


def test_an_all_negative_fold_still_scores_zero():
    proba = [np.array([[1.0], [1.0]])]
    out = cc.positive_proba(_Wrapper(proba, [_SingleClassEstimator(0)]), X=None)
    assert out.ravel() == pytest.approx([0.0, 0.0])


def test_labels_are_resolved_independently_of_one_another():
    """Each label has its own estimator, so a single-class label must not take its reading from a
    neighbouring one."""
    proba = [np.array([[1.0], [1.0]]), np.array([[1.0], [1.0]])]
    estimators = [_SingleClassEstimator(1), _SingleClassEstimator(0)]
    out = cc.positive_proba(_Wrapper(proba, estimators), X=None)
    assert out[:, 0] == pytest.approx([1.0, 1.0])
    assert out[:, 1] == pytest.approx([0.0, 0.0])


def test_a_2d_proba_is_passed_straight_through():
    """XGBoost's native multilabel output is already (n_samples, n_labels) of P(label==1)."""
    class _Native:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])

    assert cc.positive_proba(_Native(), X=None).tolist() == [[0.1, 0.9]]


def test_the_all_positive_fold_no_longer_maximises_the_loss():
    """What the bug did downstream: cross-entropy against an all-positive label."""
    y_true = np.array([[1], [1]])
    proba = [np.array([[1.0], [1.0]])]
    scored = cc.positive_proba(_Wrapper(proba, [_SingleClassEstimator(1)]), X=None)

    assert cc.mean_cross_entropy(y_true, scored) < 1e-6
    # What the old code produced for the same fold, for contrast.
    assert cc.mean_cross_entropy(y_true, np.zeros_like(scored)) > 30


# --------------------------------------------------------------------------- grouped inner CV

def test_inner_cv_is_grouped_by_document():
    assert isinstance(cc.inner_cv(), GroupKFold)
    assert cc.inner_cv().get_n_splits() == cc.CV_FOLDS


def test_no_document_spans_a_fold_boundary():
    """The property that matters: a paper split into several segments must land wholly in train or
    wholly in validation. Segments of one paper are near-duplicate text carrying an identical
    label, so a split that separates them scores the model on text it has already seen."""
    # 9 documents, three of which occupy several rows.
    stems = ["a", "a", "a", "b", "c", "d", "d", "e", "f", "g", "h", "h", "i"]
    X = np.arange(len(stems)).reshape(-1, 1)
    y = np.zeros(len(stems))

    for train_idx, test_idx in cc.inner_cv().split(X, y, groups=stems):
        train_stems = {stems[i] for i in train_idx}
        test_stems = {stems[i] for i in test_idx}
        assert train_stems.isdisjoint(test_stems)


def test_an_ungrouped_split_would_have_leaked():
    """Contrast, so the test above is not vacuous: the default row-wise KFold does split a
    multi-segment document across the boundary for this arrangement."""
    from sklearn.model_selection import KFold

    # 12 rows over 3 folds splits [0-3] [4-7] [8-11], so rows 3 and 4 straddle the first boundary.
    stems = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    stems[4] = stems[3]  # one document occupying rows 3 and 4, either side of that boundary
    X = np.arange(len(stems)).reshape(-1, 1)

    leaked = any(
        not {stems[i] for i in tr}.isdisjoint({stems[i] for i in te})
        for tr, te in KFold(n_splits=cc.CV_FOLDS).split(X)
    )
    assert leaked


# ------------------------------------------- deduping embedding work must not drop dataset types

def test_dedupe_keeps_one_unit_per_hash():
    """Embedding the same text once per dataset would be three identical paid calls for one vector."""
    units = [
        ("h1", "WPAuthorClf::raw::full", "same text"),
        ("h1", "WPAuthorClf::naive::full", "same text"),
        ("h1", "WPAuthorClf::llm_censorship::full", "same text"),
    ]
    unique, _ = cc.collect_unique_units(units)
    assert list(unique) == ["h1"]
    assert unique["h1"][1] == "WPAuthorClf::raw::full", "first wins, and DATASETS is raw-first"


def test_dedupe_remembers_every_type_a_hash_appeared_under():
    """The fix. A plain setdefault kept only the winning tuple, so the other two type strings were
    never submitted to the embedder -- and `get_or_generate_embedding`'s type accumulation, which
    exists for exactly this, never saw them."""
    units = [
        ("h1", "WPAuthorClf::raw::full", "same text"),
        ("h1", "WPAuthorClf::naive::full", "same text"),
        ("h1", "WPAuthorClf::llm_censorship::full", "same text"),
    ]
    _, seen = cc.collect_unique_units(units)
    assert seen["h1"] == {
        "WPAuthorClf::raw::full",
        "WPAuthorClf::naive::full",
        "WPAuthorClf::llm_censorship::full",
    }


def test_documents_censorship_changed_keep_distinct_hashes():
    """The contrasting case: censorship altered the text, so each dataset has its own hash and
    there is nothing to reconcile."""
    units = [
        ("h_raw", "WPAuthorClf::raw::full", "mentions Norway"),
        ("h_naive", "WPAuthorClf::naive::full", "mentions CountryName"),
    ]
    unique, seen = cc.collect_unique_units(units)
    assert set(unique) == {"h_raw", "h_naive"}
    assert seen["h_raw"] == {"WPAuthorClf::raw::full"}
    assert seen["h_naive"] == {"WPAuthorClf::naive::full"}


# ----------------------------------------- cached artefacts must be keyed on the data, not the slug

def _plan(rows):
    """A hash_labels plan: (hash, label vector, stem) per row."""
    return [(h, np.array(label), stem) for h, label, stem in rows]


BASE_PLAN = _plan([
    ("hash_a", [1, 0], "ATCM10_wp001_e"),
    ("hash_b", [0, 1], "ATCM10_wp002_e"),
])


def test_fingerprint_is_stable_for_the_same_dataset():
    assert cc.dataset_fingerprint(BASE_PLAN) == cc.dataset_fingerprint(BASE_PLAN)


def test_fingerprint_ignores_row_order():
    """It describes the dataset's content, so enumeration order must not change it."""
    assert cc.dataset_fingerprint(BASE_PLAN) == cc.dataset_fingerprint(list(reversed(BASE_PLAN)))


def test_fingerprint_changes_when_the_text_changes():
    """Row hashes are sha256 of the *censored* text, so a re-OCR, a re-embed or a censorship fix
    all move them -- which is exactly the case where reusing a cached model reports old numbers."""
    changed = _plan([("hash_a_v2", [1, 0], "ATCM10_wp001_e"), ("hash_b", [0, 1], "ATCM10_wp002_e")])
    assert cc.dataset_fingerprint(changed) != cc.dataset_fingerprint(BASE_PLAN)


def test_fingerprint_changes_when_a_label_changes():
    relabelled = _plan([("hash_a", [0, 0], "ATCM10_wp001_e"), ("hash_b", [0, 1], "ATCM10_wp002_e")])
    assert cc.dataset_fingerprint(relabelled) != cc.dataset_fingerprint(BASE_PLAN)


def test_fingerprint_changes_when_a_row_is_added_or_removed():
    bigger = BASE_PLAN + _plan([("hash_c", [1, 1], "ATCM10_wp003_e")])
    assert cc.dataset_fingerprint(bigger) != cc.dataset_fingerprint(BASE_PLAN)
    assert cc.dataset_fingerprint(BASE_PLAN[:1]) != cc.dataset_fingerprint(BASE_PLAN)


def test_fingerprint_changes_when_the_split_moves():
    """Same rows, reassigned between train and val: the cached models no longer describe it."""
    a, b = BASE_PLAN[:1], BASE_PLAN[1:]
    assert cc.dataset_fingerprint(a, b) == cc.dataset_fingerprint(BASE_PLAN)
    moved = _plan([("hash_a", [1, 0], "OTHER_STEM")])
    assert cc.dataset_fingerprint(moved, b) != cc.dataset_fingerprint(BASE_PLAN)


def test_cache_is_reusable_when_no_fingerprint_was_recorded(tmp_path):
    """Absence of evidence is not evidence of staleness -- a first run, or a cache predating this
    check, must not be thrown away."""
    assert cc.cached_artefacts_are_current("raw__full", "abc", tmp_path) is True


def test_cache_is_reusable_when_the_fingerprint_matches(tmp_path):
    cc.write_dataset_fingerprint("raw__full", "abc", tmp_path)
    assert cc.cached_artefacts_are_current("raw__full", "abc", tmp_path) is True


def test_cache_is_rejected_when_the_fingerprint_differs(tmp_path):
    cc.write_dataset_fingerprint("raw__full", "abc", tmp_path)
    assert cc.cached_artefacts_are_current("raw__full", "xyz", tmp_path) is False


def test_fingerprints_are_per_dataset(tmp_path):
    cc.write_dataset_fingerprint("raw__full", "abc", tmp_path)
    assert cc.cached_artefacts_are_current("naive__full", "abc", tmp_path) is True
