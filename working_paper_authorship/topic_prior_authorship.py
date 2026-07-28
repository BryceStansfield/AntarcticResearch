"""Topic-conditioned Beta-Bernoulli authorship priors, transferred from Working Papers to Measures.

The question
------------
``not_effective_measure_authorship.py`` showed that the *discriminative* WP authorship
classifiers do not transfer to the not-yet-effective measures — every model scores far worse
than the no-skill baseline (BCE 1.46-2.01 against a 0.30 base-rate floor). That is a strong
OOD negative result for models that read the document's own text.

This asks a much weaker question with a much simpler estimator: forget the text of the measure
entirely, and ask only **"which countries author working papers about this kind of thing?"**
If subject matter alone carries country signal that survives the WP -> Measure shift, a
topic-conditioned base rate should beat an unconditioned one. If it does not, the topic adds
nothing over "how often does this country show up at all".

The estimator
-------------
1. Fit one BERTopic model over the *combined* space of every instrument and every English
   Working Paper (``fit_combined_topic_model``) — the same model ``measure_wp_topics.py``
   builds, so a topic is a shared concept rather than a per-corpus artefact. All 52 evaluation
   measures live in ``MeasureCorpusEnriched.csv`` and are embedded, so they are clustered along
   with everything else; nothing is held out of the *clustering*.

2. Keep topics that are not the outlier topic (-1) and contain at least one Working Paper.
   A topic with no WP has no evidence to form a posterior from.

3. Within topic ``k``, treat each country ``c``'s authorship as independent Bernoulli with a
   Beta(1, 1) prior. Observing the ``n_k`` Working Papers in that topic, of which ``s_kc`` list
   ``c`` as an author, gives

       p_kc | data  ~  Beta(1 + s_kc, 1 + n_k - s_kc)

   and the posterior mean ``(1 + s_kc) / (2 + n_k)`` is the prediction for *every* measure in
   topic ``k``. This is Laplace-smoothed counting; the prior matters most in small topics,
   which is the point of using one.

4. **Working Papers only** contribute to the posterior. Measure labels are never read during
   fitting — they are the evaluation target and nothing else.

The target
----------
``data/Not-Effective measures.csv``: for each measure, the ``TRUE`` column lists the Parties
that ratified it and ``FALSE`` those that did not. Ratification stands in for authorship. This
is a genuine distributional shift from WP authorship (ratifying is cheaper and far more common
— 78% of labelled cells are positive, against a much sparser WP authorship rate), and it is
the reason the 52 measures are evaluation-only: 52 documents cannot train anything.

Country universe: the same five the discriminative classifiers use (Australia, United Kingdom,
United States, Norway, Chile). The evaluation CSV labels 30 countries and a Beta-Bernoulli
posterior needs no training, so all 30 are scoreable — but the five are by a wide margin the
most prolific WP authors, which puts their transferred posteriors closest to the ratification
rate and so leaves the least calibration gap for BCE to punish. Keeping the same five also
makes the numbers directly readable against
``data/not_effective_measure_predictions/report.txt``.

Baselines (all computed on the same measures/cells as the model, so BCE is comparable)
-------------------------------------------------------------------------------------
* ``uniform``        — 0.5 everywhere. BCE = ln 2 = 0.6931. The prior with no data at all.
* ``global_wp``      — the same Beta(1,1) posterior pooled over *all* Working Papers, ignoring
                       topic: ``(1 + s_c) / (2 + N)``. **This is the comparison that answers the
                       question.** Topic-conditioning is only informative if it beats this.
* ``measure_oracle`` — each country's base rate computed on the evaluation labels themselves.
                       Cheating by construction and unbeatable by any honest predictor; it is
                       the floor that says how much of the target is just per-country prevalence.

Measures landing in an ineligible topic (outlier, or a topic with no WP) have no topic
posterior. Both handlings are reported: ``eligible-only`` restricts every method to the
measures that got one, and ``with-fallback`` covers all 52 by backing off to ``global_wp``.

Outputs (to ``data/topic_prior_authorship/``):
  * ``cell_predictions.csv``  — one row per (measure, country): topic, posterior, truth
  * ``topic_posteriors.csv``  — one row per eligible topic per country: n_k, s_kc, posterior
  * ``report.txt``            — the BCE comparison table
"""

import pathlib

import numpy as np
import pandas as pd

from adhoc_analyses.measure_wp_topics import (
    fit_combined_topic_model,
    load_measures,
    load_working_papers,
)
from country_meta_info import alternative_names_to_countries
from utils import split_parties

MEASURES_CSV = pathlib.Path("data/Not-Effective measures.csv")
OUTPUT_DIR = pathlib.Path("data/topic_prior_authorship")

# The five the discriminative classifiers are trained on. Everything below is restricted to
# these: they dominate WP authorship, so their transferred posteriors need the least
# recalibration, and the results stay comparable with the classifiers' report.
CLASSIFIER_COUNTRIES = ["Australia", "United Kingdom", "United States", "Norway", "Chile"]

# Beta(a, b) prior on every country's per-topic authorship probability. (1, 1) is uniform on
# [0, 1], so the posterior mean is Laplace's rule of succession: (1 + s) / (2 + n).
PRIOR_A = 1.0
PRIOR_B = 1.0

# log_loss-style clipping, so an oracle base rate of exactly 0 or 1 cannot produce an infinite
# BCE. The Beta posteriors are strictly interior already and are unaffected.
EPS = 1e-15


def canonical(name: str) -> str:
    """Fold a country string to one lowercase canonical form.

    Both sides need this and they disagree: the evaluation CSV says "Russian Federation" and
    "Korea (ROK)" where ``country_meta_info`` canonicalises to "Russia" and "Republic of Korea".
    ``split_parties`` already lowercases the WP side, so everything is lowered here too.
    """
    stripped = str(name).strip()
    return str(alternative_names_to_countries.get(stripped, stripped)).strip().lower()


def parse_country_cell(cell) -> set[str]:
    """Comma-separated country list -> canonical names."""
    if pd.isna(cell):
        return set()
    return {canonical(part) for part in str(cell).split(",") if part.strip()}


def wp_author_countries(working_paper: dict) -> set[str]:
    """Canonical author countries of one Working Paper.

    Non-party authors (SCAR, ASOC, IAATO, ...) simply canonicalise to themselves and never match
    a target country, so they drop out of the counts without needing an explicit filter.
    """
    parties = working_paper.get("parties")
    if parties is None:
        return set()
    return {canonical(p) for p in split_parties([str(p) for p in list(parties)])}


def load_evaluation_labels(countries: list[str]) -> pd.DataFrame:
    """One row per (measure, country) with a 0/1 ratification label, for ``countries`` only.

    Long format rather than a matrix so the join to topics stays explicit. Every measure names
    all five target countries across TRUE/FALSE, so the result is rectangular; a country named
    in neither would be unlabelled rather than negative, and is raised on rather than assumed.
    """
    df = pd.read_csv(MEASURES_CSV)
    wanted = set(countries)
    rows = []
    for row in df.itertuples():
        ratified = parse_country_cell(row.TRUE)
        not_ratified = parse_country_cell(getattr(row, "FALSE"))
        overlap = ratified & not_ratified
        if overlap:
            raise ValueError(
                f"Measure {row.Document_Number}: {sorted(overlap)} appear in both TRUE and "
                f"FALSE — the label is contradictory."
            )
        missing = wanted - (ratified | not_ratified)
        if missing:
            raise ValueError(
                f"Measure {row.Document_Number}: {sorted(missing)} appear in neither TRUE nor "
                f"FALSE — cannot assign a ground-truth label."
            )
        for country in sorted(wanted):
            rows.append(
                {
                    "measure_id": int(row.Document_Number),
                    "country": country,
                    "label": int(country in ratified),
                }
            )
    return pd.DataFrame(rows)


def topic_posteriors(topics, docs, countries: list[str]) -> tuple[pd.DataFrame, dict]:
    """Beta-Bernoulli posterior mean per (eligible topic, country), from Working Papers only.

    Returns the long table and a ``{topic: {country: p}}`` lookup. Eligible means: not the
    outlier topic, and at least one Working Paper to form a posterior from.
    """
    wp_authors: dict[int, list[set[str]]] = {}
    for topic, doc in zip(topics, docs):
        if doc["doc_class"] != "WorkingPaper":
            continue
        wp_authors.setdefault(int(topic), []).append(wp_author_countries(doc))

    rows = []
    lookup: dict[int, dict[str, float]] = {}
    for topic, author_sets in sorted(wp_authors.items()):
        if topic == -1:
            continue  # outlier "topic" is a leftover bin, not a concept
        n_k = len(author_sets)
        lookup[topic] = {}
        for country in countries:
            s_kc = sum(country in authors for authors in author_sets)
            posterior = (PRIOR_A + s_kc) / (PRIOR_A + PRIOR_B + n_k)
            lookup[topic][country] = posterior
            rows.append(
                {
                    "topic": topic,
                    "country": country,
                    "n_working_papers": n_k,
                    "n_authored": s_kc,
                    "posterior_mean": posterior,
                }
            )
    return pd.DataFrame(rows), lookup


def global_posterior(topics, docs, countries: list[str]) -> dict[str, float]:
    """The same Beta(1,1) posterior pooled over every Working Paper, ignoring topic.

    Pooled over the identical WP set the per-topic posteriors are built from (outlier-topic
    papers included), so the only difference between this and the topic model is the
    conditioning — which is what makes the BCE gap interpretable.
    """
    author_sets = [
        wp_author_countries(doc) for doc in docs if doc["doc_class"] == "WorkingPaper"
    ]
    n = len(author_sets)
    return {
        c: (PRIOR_A + sum(c in authors for authors in author_sets)) / (PRIOR_A + PRIOR_B + n)
        for c in countries
    }


def binary_cross_entropy(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean BCE over cells (nats).

    A flat mean over (measure, country) cells rather than a mean of per-country losses, because
    the labels are ragged — a per-country mean would silently reweight countries by how many
    measures happen to mention them. The per-country breakdown is reported separately.
    """
    p = np.clip(probs, EPS, 1 - EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def evaluate(cells: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """BCE per method over a fixed set of cells, plus accuracy at a 0.5 threshold."""
    labels = cells["label"].to_numpy(dtype=float)
    rows = []
    for method in methods:
        probs = cells[method].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "n_cells": len(cells),
                "bce": binary_cross_entropy(labels, probs),
                "accuracy": float(((probs >= 0.5).astype(float) == labels).mean()),
                "mean_prob": float(probs.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("bce")


def per_country_bce(cells: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for country, group in cells.groupby("country"):
        row = {
            "country": country,
            "n_cells": len(group),
            "positive_rate": float(group["label"].mean()),
        }
        for method in methods:
            row[method] = binary_cross_entropy(
                group["label"].to_numpy(dtype=float), group[method].to_numpy(dtype=float)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("country")


def discrimination_test(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Does the topic posterior say anything *beyond* each country's own prevalence?

    Raw BCE conflates two failures. Calibration: WP authorship runs a few percent per country
    while ratification runs 78%, so a posterior transferred verbatim is bound to look terrible
    no matter how good its ordering is. Discrimination: whether the posterior ranks the right
    measures higher *within* a country. Only the second is the research question — a constant
    offset would fix the first.

    So, per country, fit a 1-D logistic regression of the label on ``logit(topic_posterior)``
    and compare it to the intercept-only model (which is exactly ``measure_oracle``, that
    country's prevalence). The two are nested, so 2*(difference in log-likelihood) is a
    likelihood-ratio chi-square on 1 df, and summing over countries gives a joint test. Both
    fits use the evaluation labels, so these are oracle-calibrated upper bounds on what the
    topic posterior could deliver — the honest reading is "even given free recalibration,
    is there signal?"

    Countries whose labels are all-0 or all-1 among these measures are skipped: prevalence
    already predicts them perfectly and the logistic fit is degenerate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy import stats

    rows = []
    for country, group in cells.groupby("country"):
        y = group["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2 or len(y) < 8:
            continue
        p = np.clip(group["topic_posterior"].to_numpy(dtype=float), EPS, 1 - EPS)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        if np.ptp(x) == 0:
            continue  # every measure in one topic: nothing to rank

        # Unpenalised, so the comparison against the intercept-only model is a clean LR test.
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(x, y)
        fitted = np.clip(model.predict_proba(x)[:, 1], EPS, 1 - EPS)
        ll_full = float(np.sum(y * np.log(fitted) + (1 - y) * np.log(1 - fitted)))

        base = np.clip(np.full_like(fitted, y.mean()), EPS, 1 - EPS)
        ll_null = float(np.sum(y * np.log(base) + (1 - y) * np.log(1 - base)))

        stat = 2 * (ll_full - ll_null)
        rows.append(
            {
                "country": country,
                "n": len(y),
                "positive_rate": float(y.mean()),
                # AUC of the raw posterior: calibration-free by construction.
                "auc": float(roc_auc_score(y, p)),
                "slope": float(model.coef_[0][0]),
                "lr_chi2": stat,
                "p_value": float(stats.chi2.sf(max(stat, 0.0), df=1)),
                "bce_recalibrated": binary_cross_entropy(y.astype(float), fitted),
                "bce_prevalence_only": binary_cross_entropy(y.astype(float), base),
            }
        )

    table = pd.DataFrame(rows).sort_values("p_value")
    joint = {
        "n_countries": len(table),
        "chi2": float(table["lr_chi2"].sum()) if len(table) else 0.0,
        "df": len(table),
        "mean_auc": float(table["auc"].mean()) if len(table) else float("nan"),
    }
    joint["p_value"] = float(stats.chi2.sf(joint["chi2"], df=joint["df"])) if len(table) else 1.0
    return table, joint


def build_cells(labels: pd.DataFrame, measure_topic: dict[int, int],
                posteriors: dict, global_p: dict[str, float]) -> pd.DataFrame:
    """Join every (measure, country) label to its topic posterior and the baselines."""
    cells = labels.copy()
    cells["topic"] = cells["measure_id"].map(measure_topic)
    # A measure is eligible when its topic formed a posterior at all.
    cells["eligible"] = cells["topic"].map(lambda t: t in posteriors).fillna(False)

    cells["topic_posterior"] = [
        posteriors[t][c] if eligible else np.nan
        for t, c, eligible in zip(cells["topic"], cells["country"], cells["eligible"])
    ]
    cells["global_wp"] = cells["country"].map(global_p)
    cells["uniform"] = 0.5
    # Oracle: per-country prevalence read off the evaluation labels themselves.
    cells["measure_oracle"] = cells["country"].map(cells.groupby("country")["label"].mean())
    # Topic posterior where one exists, unconditioned WP posterior elsewhere.
    cells["topic_with_fallback"] = cells["topic_posterior"].fillna(cells["global_wp"])
    return cells


def _table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda x: f"{x:.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached embeddings...")
    measures = load_measures()
    working_papers = load_working_papers()
    docs = measures + working_papers
    print(f"  {len(measures)} instruments, {len(working_papers)} English working papers")

    countries = sorted({canonical(c) for c in CLASSIFIER_COUNTRIES})
    labels = load_evaluation_labels(countries)
    print(f"  evaluation: {labels['measure_id'].nunique()} measures x {len(countries)} countries "
          f"= {len(labels)} labelled cells ({labels['label'].mean():.1%} positive)")

    print("\nFitting the combined topic model over instruments + working papers...")
    topic_model, topics = fit_combined_topic_model(docs)

    posterior_table, posteriors = topic_posteriors(topics, docs, countries)
    posterior_table.to_csv(OUTPUT_DIR / "topic_posteriors.csv", index=False)
    n_topics = len({int(t) for t in topics if int(t) != -1})
    print(f"  {n_topics} topics; {len(posteriors)} of them contain >=1 working paper and are eligible")

    global_p = global_posterior(topics, docs, countries)

    # Instruments carry uuid "MEASURE__<Document_Number>", which is the evaluation CSV's key.
    measure_topic = {
        int(doc["uuid"].removeprefix("MEASURE__")): int(topic)
        for topic, doc in zip(topics, docs)
        if doc["doc_class"] == "Instrument"
    }

    cells = build_cells(labels, measure_topic, posteriors, global_p)
    cells.to_csv(OUTPUT_DIR / "cell_predictions.csv", index=False)

    eligible = cells[cells["eligible"]]
    n_eligible = eligible["measure_id"].nunique()
    n_total = cells["measure_id"].nunique()
    print(f"  {n_eligible} of {n_total} evaluation measures landed in an eligible topic")

    lines = [
        "TOPIC-CONDITIONED BETA-BERNOULLI AUTHORSHIP PRIOR, WP -> MEASURE TRANSFER",
        f"Prior: Beta({PRIOR_A:g}, {PRIOR_B:g}) per (topic, country); posterior from WORKING PAPERS ONLY.",
        f"Target: {MEASURES_CSV} ratification (TRUE=ratified=1, FALSE=not=0), evaluation-only.",
        f"{n_total} measures x {len(countries)} countries = {len(cells)} cells, "
        f"{cells['label'].mean():.1%} positive.",
        f"Topics: {n_topics} non-outlier, {len(posteriors)} eligible (>=1 WP). "
        f"{n_eligible}/{n_total} measures landed in one.",
        "Binary cross-entropy in nats, lower is better. uniform = ln2 = 0.6931.",
        "",
        f"--- eligible-topic measures only ({n_eligible} measures, {len(eligible)} cells) ---",
    ]

    eligible_methods = ["topic_posterior", "global_wp", "uniform", "measure_oracle"]
    eligible_scores = evaluate(eligible, eligible_methods)
    lines += [_table(eligible_scores), ""]

    lines += [f"--- all measures, ineligible backed off to global_wp ({n_total} measures, {len(cells)} cells) ---"]
    all_methods = ["topic_with_fallback", "global_wp", "uniform", "measure_oracle"]
    lines += [_table(evaluate(cells, all_methods)), ""]

    # The headline: topic-conditioning is only informative if it beats the unconditioned posterior.
    topic_bce = float(eligible_scores.loc[eligible_scores["method"] == "topic_posterior", "bce"].iloc[0])
    global_bce = float(eligible_scores.loc[eligible_scores["method"] == "global_wp", "bce"].iloc[0])
    delta = topic_bce - global_bce
    verdict = "BETTER than" if delta < 0 else "WORSE than"
    lines += [
        f"HEADLINE: topic-conditioned BCE {topic_bce:.4f} vs unconditioned WP posterior "
        f"{global_bce:.4f} ({delta:+.4f}, {verdict} unconditioned).",
        "",
        f"--- per-country BCE (eligible-topic measures) ---",
        _table(per_country_bce(eligible, eligible_methods)),
        "",
    ]

    # Raw BCE punishes the WP->ratification prevalence gap and the ranking together. This
    # separates them, so a bad BCE with real ranking signal cannot be mistaken for no signal.
    disc, joint = discrimination_test(eligible)
    disc.to_csv(OUTPUT_DIR / "discrimination_test.csv", index=False)
    lines += [
        "--- discrimination, with calibration given away for free (eligible-topic measures) ---",
        "Per country: logistic fit of the label on logit(topic_posterior), against the",
        "intercept-only model (= that country's prevalence = measure_oracle). Nested, so the",
        "likelihood-ratio chi2 on 1 df asks whether the topic posterior adds anything at all.",
        "Both fits see the evaluation labels, so these are ORACLE-CALIBRATED UPPER BOUNDS.",
        f"Joint over {joint['n_countries']} testable countries: chi2 = {joint['chi2']:.2f} on "
        f"{joint['df']} df, p = {joint['p_value']:.4f}; mean AUC = {joint['mean_auc']:.4f}.",
        _table(disc),
        "",
    ]

    lines += [
        "Read against data/not_effective_measure_predictions/report.txt, which scores the same",
        "52 measures and the same five countries: the discriminative WP classifiers there ran",
        "BCE 1.46-2.01 against the identical 0.2992 measure_oracle floor.",
    ]

    report = "\n".join(lines)
    (OUTPUT_DIR / "report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
