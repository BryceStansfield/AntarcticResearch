"""Compare closest-WP match distance for Annex-V vs non-Annex-V measures.

Annex-V measures are identified by an empty `Approvals` column (no ratification
date list). Uses the same embedding / nearest-neighbour path as the real
MeasureWPIntroducers pipeline.
"""
import numpy as np
import pandas as pd

import embeddings.document_embeddings as de
from antarctic_ladder_metrics.constants import START_YEAR, END_YEAR

# Annex-V lookup: Document_Number -> is a Fast Approval measure (Annex-V measures
# use the Article 6/8 fast-track approval, flagged in the Status column).
df = pd.read_csv("data/MeasureCorpusEnriched.csv")
fast_approval = df["Status"].fillna("").astype(str).str.contains("Fast Approval", case=False)
annexv_by_docnum = {str(dn): bool(e) for dn, e in zip(df["Document_Number"], fast_approval)}

document_getter = de.DocumentTextGetter()
measures = [m for m in document_getter.get_all_of_type("measure")
            if START_YEAR <= m["year"] <= END_YEAR]

wp_getter = de.EmbeddingLookerUpper("WorkingPaper")

records = []
for m in measures:
    nearest = wp_getter.get_nearest_neighbours(m["uuid"], 1)  # [(wp_uuid, cosine_dist)]
    closest_dist = nearest[0][1]
    is_annexv = annexv_by_docnum.get(str(m["measure_id"]))
    records.append({"measure_id": m["measure_id"], "year": m["year"],
                    "is_annexv": is_annexv, "closest_dist": closest_dist})

res = pd.DataFrame(records)
n_missing = res["is_annexv"].isna().sum()
res = res.dropna(subset=["is_annexv"])
res["is_annexv"] = res["is_annexv"].astype(bool)

def describe(s):
    return (f"n={len(s):4d}  mean={s.mean():.4f}  median={s.median():.4f}  "
            f"std={s.std():.4f}  min={s.min():.4f}  "
            f"q25={s.quantile(.25):.4f}  q75={s.quantile(.75):.4f}  max={s.max():.4f}")

annexv = res.loc[res["is_annexv"], "closest_dist"]
other = res.loc[~res["is_annexv"], "closest_dist"]

print(f"Measures analysed: {len(res)}  (dropped {n_missing} with no CSV match)")
print(f"  Annex-V (Fast Approval):       {describe(annexv)}")
print(f"  non-Annex-V (other):           {describe(other)}")
print(f"\n  mean difference (Annex-V - other): {annexv.mean() - other.mean():+.4f}")

try:
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(annexv, other, alternative="two-sided")
    print(f"  Mann-Whitney U two-sided p-value: {p:.3e}")
except ImportError:
    print("  (scipy not available - skipping significance test)")
