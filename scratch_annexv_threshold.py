"""Distance-threshold sensitivity for Annex-V vs non-Annex-V measure matching.

For a sweep of cosine-distance cutoffs, report how many measures in each group
would have their closest WP match dropped (closest_dist > threshold).
"""
import numpy as np
import pandas as pd

import embeddings.document_embeddings as de
from antarctic_ladder_metrics.constants import START_YEAR, END_YEAR

df = pd.read_csv("data/MeasureCorpusEnriched.csv")
fast_approval = df["Status"].fillna("").astype(str).str.contains("Fast Approval", case=False)
annexv_by_docnum = {str(dn): bool(e) for dn, e in zip(df["Document_Number"], fast_approval)}

document_getter = de.DocumentTextGetter()
measures = [m for m in document_getter.get_all_of_type("measure")
            if START_YEAR <= m["year"] <= END_YEAR]
wp_getter = de.EmbeddingLookerUpper("WorkingPaper")

records = []
for m in measures:
    nearest = wp_getter.get_nearest_neighbours(m["uuid"], 1)
    records.append({"is_annexv": annexv_by_docnum.get(str(m["measure_id"])),
                    "closest_dist": nearest[0][1]})

res = pd.DataFrame(records).dropna(subset=["is_annexv"])
res["is_annexv"] = res["is_annexv"].astype(bool)

annexv = res.loc[res["is_annexv"], "closest_dist"]
other = res.loc[~res["is_annexv"], "closest_dist"]
n_a, n_o = len(annexv), len(other)

# Candidate thresholds: percentiles of the non-Annex-V (well-matched) group,
# plus a few round values.
candidates = sorted(set(
    [round(other.quantile(q), 3) for q in (0.75, 0.90, 0.95, 0.99)]
    + [0.13, 0.15, 0.18, 0.20, 0.25]
))

print(f"Groups: Annex-V n={n_a}, non-Annex-V n={n_o}\n")
print(f"{'threshold':>9} | {'Annex-V dropped':>22} | {'non-Annex-V dropped':>22} | {'drop-rate ratio':>15}")
print("-" * 80)
for t in candidates:
    da, do = (annexv > t).sum(), (other > t).sum()
    pa, po = da / n_a, do / n_o
    ratio = (pa / po) if po > 0 else float("inf")
    note = ""
    if abs(t - round(other.quantile(0.75), 3)) < 1e-9:
        note = "  <- non-Annex-V q75"
    print(f"{t:9.3f} | {da:5d} ({pa:5.1%}){'':>9} | {do:5d} ({po:5.1%}){'':>9} | {ratio:14.2f}x{note}")
