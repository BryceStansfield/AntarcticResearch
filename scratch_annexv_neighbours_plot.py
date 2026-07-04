"""Plot the distribution of the first N nearest-WP distances for Annex-V vs
non-Annex-V measures (per-rank mean with IQR band)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import embeddings.document_embeddings as de
from antarctic_ladder_metrics.constants import START_YEAR, END_YEAR

N = 20

df = pd.read_csv("data/MeasureCorpusEnriched.csv")
fast_approval = df["Status"].fillna("").astype(str).str.contains("Fast Approval", case=False)
annexv_by_docnum = {str(dn): bool(e) for dn, e in zip(df["Document_Number"], fast_approval)}

document_getter = de.DocumentTextGetter()
measures = [m for m in document_getter.get_all_of_type("measure")
            if START_YEAR <= m["year"] <= END_YEAR]
wp_getter = de.EmbeddingLookerUpper("WorkingPaper")

annexv_rows, other_rows = [], []
for m in measures:
    nearest = wp_getter.get_nearest_neighbours(m["uuid"], N)
    dists = [d for _, d in nearest]
    flag = annexv_by_docnum.get(str(m["measure_id"]))
    if flag is None:
        continue
    (annexv_rows if flag else other_rows).append(dists)

annexv = np.array(annexv_rows)   # (n_a, N)
other = np.array(other_rows)     # (n_o, N)
ranks = np.arange(1, N + 1)

fig, ax = plt.subplots(figsize=(9, 5.5))
for data, color, label in [(annexv, "#d1495b", f"Annex-V (n={len(annexv)})"),
                           (other, "#2e86ab", f"non-Annex-V (n={len(other)})")]:
    mean = data.mean(axis=0)
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    ax.plot(ranks, mean, color=color, lw=2, marker="o", ms=4, label=label)
    ax.fill_between(ranks, q25, q75, color=color, alpha=0.15)

ax.set_xlabel("Neighbour rank (k-th nearest working paper)")
ax.set_ylabel("Cosine distance")
ax.set_title(f"Distribution of first {N} WP neighbours: Annex-V vs non-Annex-V measures\n"
             "(line = mean, band = interquartile range)")
ax.set_xticks(ranks)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

out = "annexv_neighbours.png"
fig.savefig(out, dpi=130)
print("saved:", out)

# Also print the per-rank means for reference.
print("\nrank |  Annex-V mean | non-Annex-V mean | diff")
for k in range(N):
    a, o = annexv[:, k].mean(), other[:, k].mean()
    print(f"{k+1:4d} |   {a:.4f}     |    {o:.4f}       | {a-o:+.4f}")
