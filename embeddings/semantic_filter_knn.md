# Cross-validated kNN separability of the semantic-importance filter

**Question.** The semantic-relevance filter in [`working_paper_semantic_filter.py`](working_paper_semantic_filter.py)
labels each censored working-paper sentence IMPORTANT (carries a diplomatic stance) or FLUFF
(procedural boilerplate) with a per-sentence LLM call. Could that label instead be recovered from
the sentence *embeddings* — i.e. is it a candidate for a cheaper semi-supervised approach? This is
the quantitative companion to the UMAP projections in
[`semantic_filter_umap.py`](semantic_filter_umap.py): rather than eyeballing a 2D/3D picture, we
ask directly whether a sentence's nearest neighbours in embedding space share its label.

**Method.** Cross-validated k-nearest-neighbours, `k=5`, **cosine** distance, on the raw
4096-dim `qwen/qwen3-embedding-8b` embeddings. Stratified 5-fold CV (shuffled, seed 42), so every
sentence is predicted from a fold in which it did not participate. Reproduce with:

```
python -m embeddings.semantic_filter_knn
```

**Data.** Every censored sentence that has *both* a cached importance label and a cached
embedding, obtained by an equijoin — the importance cache keys on `sha256(sentence)` and the
embedder keys each unit on `sha256(text)`, so `sentence_hash == document_uuid` pairs each label
with the embedding of the byte-identical sentence (no re-embedding). After embedding every paper's
LLM-censored sentences (not just the authorship classifier's 5-country subset — see
`embed_all_llm_censored_working_paper_sentences` in
[`embed_all_documents.py`](embed_all_documents.py)) this is **128,798 sentences: 39,534 IMPORTANT /
89,264 FLUFF** (fluff base rate 69.3%) — i.e. all 128,801 labelled sentences bar 3 long-sentence
sub-split stragglers.

## Results

| Metric | kNN (k=5, cosine) | Baseline |
|---|---:|---:|
| Accuracy | **0.836** | 0.693 (predict majority class) |
| Balanced accuracy | **0.812** | 0.500 |
| ROC-AUC | **0.897** | 0.500 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| IMPORTANT | 0.725 | 0.751 | 0.738 |
| FLUFF | 0.888 | 0.874 | 0.881 |

Confusion matrix (rows = true, columns = predicted; order `[FLUFF, IMPORTANT]`):

```
            pred FLUFF   pred IMPORTANT
true FLUFF      78001         11263
true IMPORTANT   9844         29690
```

The result is stable against corpus size: on the earlier target-country-only half (67,535
sentences) the numbers were accuracy 0.835, balanced accuracy 0.823, AUC 0.900, IMPORTANT recall
0.787. Extending to all papers left accuracy and AUC essentially unchanged and nudged IMPORTANT
recall down ~3.5 points (0.787 → 0.751), consistent with the fluff base rate rising from 67.3% to
69.3% and non-target papers adding somewhat harder IMPORTANT cases. The conclusion does not depend
on the country subset.

## Interpretation

The embedding neighbourhood carries the importance label: a sentence's 5 nearest cosine
neighbours predict its LLM label ~84% of the time — +14 points over the majority-class baseline,
balanced accuracy 0.81, AUC 0.90. This is consistent with, and quantifies, the partial
class separation already visible in the UMAP projections: importance is clearly not randomly
distributed in embedding space, but neither is it cleanly separable — the classes share a broad
overlap region, which is exactly the ~17% of neighbourhoods the classifier gets wrong.

For a semi-supervised filter this is a promising but not turnkey signal. The weakest corner is
**IMPORTANT recall (0.751)**: ~25% of important sentences sit in fluff-dominated neighbourhoods
and would be dropped. For a filter whose purpose is to *retain* substance, that is the expensive
error direction, and it is the number to watch (and tune) rather than headline accuracy. It is
tunable without new labels — a larger `k` with a probability threshold, or distance-weighted
voting — trading FLUFF precision to recover missed IMPORTANT sentences.

## Caveats

- **Sentence-level CV, not document-level.** Sentences are deduplicated by hash, but near-verbatim
  boilerplate recurs across papers, so splitting on sentences (rather than documents) can leak
  easy FLUFF neighbours between train and test and modestly flatter the FLUFF metrics. A
  document-grouped split is the stricter test.
- **Coverage is now ~100%** (128,798 of 128,801 labelled sentences; the 3 missing are long
  sentences that `split_long_document` sub-split, so they live under sub-segment hashes rather than
  `sha256(whole sentence)`). The earlier run covered only ~52% — the target-country subset the
  authorship classifier had embedded — which is why this experiment now embeds every paper's
  LLM-censored sentences up front.
- kNN is a *lower bound* on what embeddings support — a tuned classifier or label-propagation model
  would likely do better; this measures raw local geometry, not the ceiling.

## Possible next steps

- Document-grouped CV to remove the leakage concern and get an honest generalisation estimate.
- A label-propagation run from a small labelled seed, to estimate how few LLM labels would be
  needed to reproduce the filter at a target quality.
- Sweep `k` / distance-weighting / decision threshold on the IMPORTANT-recall vs FLUFF-precision
  trade-off.
