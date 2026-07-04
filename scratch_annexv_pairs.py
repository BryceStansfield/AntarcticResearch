"""Print example Annex-V (Fast Approval) measure -> closest-WP matches with a
working-paper excerpt. Shows a spread: closest, median, and farthest matches."""
import pandas as pd

import embeddings.document_embeddings as de
from antarctic_ladder_metrics.constants import START_YEAR, END_YEAR

df = pd.read_csv("data/MeasureCorpusEnriched.csv")
fast_approval = df["Status"].fillna("").astype(str).str.contains("Fast Approval", case=False)
annexv_by_docnum = {str(dn): bool(e) for dn, e in zip(df["Document_Number"], fast_approval)}
subject_by_docnum = {str(dn): s for dn, s in zip(df["Document_Number"], df["Subject"])}

document_getter = de.DocumentTextGetter()
measures = [m for m in document_getter.get_all_of_type("measure")
            if START_YEAR <= m["year"] <= END_YEAR]
wp_getter = de.EmbeddingLookerUpper("WorkingPaper")

pairs = []
for m in measures:
    if not annexv_by_docnum.get(str(m["measure_id"])):
        continue
    wp_uuid, dist = wp_getter.get_nearest_neighbours(m["uuid"], 1)[0]
    pairs.append((dist, m, wp_uuid))

pairs.sort(key=lambda x: x[0])
n = len(pairs)
# Sample a spread: 4 closest, 3 around the median, 3 farthest.
picks = pairs[:4] + pairs[n // 2 - 1: n // 2 + 2] + pairs[-3:]

for dist, m, wp_uuid in picks:
    subj = str(subject_by_docnum.get(str(m["measure_id"]), "")).strip()
    try:
        wp = document_getter.get_wp_ip_representation(wp_uuid)
    except Exception as e:
        wp = {"text": f"<lookup failed: {e}>"}
    excerpt = " ".join(str(wp.get("text", "")).split())[:420]
    print("=" * 95)
    print(f"dist={dist:.4f}   measure {m['measure_id']} ({m['year']})")
    print(f"  MEASURE subject : {subj[:160]}")
    print(f"  matched WP      : parties={list(wp.get('parties', []))}  year={wp.get('year','?')}")
    print(f"  WP excerpt      : {excerpt}")
    print()
