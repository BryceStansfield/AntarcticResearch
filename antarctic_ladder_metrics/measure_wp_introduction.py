import pandas as pd

import embeddings.document_embeddings
import utils
from antarctic_ladder_metrics.constants import *

# Non-state entities (the Secretariat, observers and coordinating bodies) that
# appear in WP author lists but should not receive country-level introduction
# credit. Stored lowercase to match utils.split_parties output.
NON_PARTY_AUTHORS = {"ats", "scar", "comnap", "iaato", "asoc", "ipy-ipo"}

class MeasureWPIntroducers():
    def __init__(self, neighbours_to_weigh = 3, candidate_neighbours = 10):
        document_getter = embeddings.document_embeddings.DocumentTextGetter()
        measures = list(filter(lambda m: START_YEAR <= m["year"] and m["year"] <= END_YEAR, document_getter.get_all_of_type("measure")))

        working_paper_getter = embeddings.document_embeddings.EmbeddingLookerUpper("WorkingPaper")

        # Keyed (measure_adoption_year, country). Credit lands in the year the measure
        # was adopted rather than the year its introducing WP was tabled.
        self.country_sums_by_year = {}
        for measure in measures:
            # Pull a wider candidate pool, then keep the nearest `neighbours_to_weigh`
            # WPs that (a) have a real (non-observer) party author and (b) predate the
            # measure. Non-party entities are stripped from each surviving author set so
            # credit goes to the actual proposing parties (avoids the measure matching
            # its own adopted text, credited to the Secretariat).
            candidates = working_paper_getter.get_nearest_neighbours(measure["uuid"], candidate_neighbours)

            kept = []
            for uuid, distance in candidates:
                representation = document_getter.get_document_representation(uuid)
                parties = [p for p in utils.split_parties(representation.get("parties", [])) if p not in NON_PARTY_AUTHORS]
                year = representation.get("year")
                if not parties or year is None or year > measure["year"]:
                    continue
                kept.append((distance, parties))
                if len(kept) == neighbours_to_weigh:
                    break

            # Inverse-distance weight across the surviving neighbours (normalised to 1).
            weight_normaliser = sum(1/distance for distance, _ in kept)
            for distance, parties in kept:
                doc_weight = (1/distance) / weight_normaliser
                for p in parties:
                    key = (measure["year"], p)
                    self.country_sums_by_year[key] = self.country_sums_by_year.get(key, 0) + doc_weight

        self.country_sums = {}
        for (_, country), weight in self.country_sums_by_year.items():
            self.country_sums[country] = self.country_sums.get(country, 0) + weight

    def country_dict(self) -> dict:
        return dict(self.country_sums)

    def figure_title(self) -> str:
        return "Measure WP Introductions"

    def save_full_figures(self, path: str):
        yearly_figures = sorted(
            ({"year": int(k[0]), "country": k[1], "value": v} for k, v in self.country_sums_by_year.items()),
            key=lambda r: (r["year"], r["country"]))
        pd.DataFrame(yearly_figures).to_csv(path)

if __name__ == "__main__":
    print(MeasureWPIntroducers().country_dict())