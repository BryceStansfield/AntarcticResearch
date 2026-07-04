"""For Annex-V measures whose closest match is an ATS self-match, show the full
top-3 neighbour set (parties + inverse-distance weight) to see how credit is
actually distributed between the Secretariat and countries."""
import pandas as pd
import utils

import embeddings.document_embeddings as de
from antarctic_ladder_metrics.constants import START_YEAR, END_YEAR

N = 3  # matches the real pipeline's neighbours_to_weigh default

df = pd.read_csv("data/MeasureCorpusEnriched.csv")
fast_approval = df["Status"].fillna("").astype(str).str.contains("Fast Approval", case=False)
annexv = {str(dn): bool(e) for dn, e in zip(df["Document_Number"], fast_approval)}
subject = {str(dn): s for dn, s in zip(df["Document_Number"], df["Subject"])}

dg = de.DocumentTextGetter()
measures = [m for m in dg.get_all_of_type("measure")
            if START_YEAR <= m["year"] <= END_YEAR]
wp = de.EmbeddingLookerUpper("WorkingPaper")

def parties_of(uuid):
    try:
        return utils.split_parties(dg.get_wp_ip_representation(uuid).get("parties", []))
    except Exception:
        return []

n_ats_top1 = 0
n_ats_only = 0            # top-1 is ATS AND no country anywhere in top-3
examples = []
for m in measures:
    if not annexv.get(str(m["measure_id"])):
        continue
    neigh = wp.get_nearest_neighbours(m["uuid"], N)
    top_parties = parties_of(neigh[0][0])
    if top_parties == ["ats"]:
        n_ats_top1 += 1
        # inverse-distance weights, normalised across the 3 (as in the pipeline)
        wsum = sum(1 / d for _, d in neigh)
        rows = [(parties_of(u), (1 / d) / wsum) for u, d in neigh]
        has_country = any(p != ["ats"] and len(p) > 0 for p, _ in rows)
        if not has_country:
            n_ats_only += 1
        if len(examples) < 6:
            examples.append((m, rows))

print(f"Annex-V measures whose CLOSEST match is an ATS self-match: {n_ats_top1}")
print(f"  ...of those, ATS-only across all top-{N} (country gets ZERO credit): {n_ats_only}")
print(f"  ...of those, a country still appears in neighbours 2-3: {n_ats_top1 - n_ats_only}\n")

for m, rows in examples:
    print("=" * 80)
    print(f"measure {m['measure_id']}: {str(subject.get(str(m['measure_id']),''))[:70]}")
    for parties, w in rows:
        print(f"    weight={w:.3f}  parties={parties}")
