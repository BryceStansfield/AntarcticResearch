import pandas as pd

import embeddings.document_embeddings
import utils
from antarctic_ladder_metrics.constants import *

# Non-state entities (the Secretariat, observers and coordinating bodies) that
# appear in WP author lists but should not receive country-level introduction
# credit. Stored lowercase to match utils.split_parties output.
NON_PARTY_AUTHORS = {"ats", "scar", "comnap", "iaato", "asoc", "ipy-ipo"}


def _select_introducing_neighbours(candidates, measure_year, neighbours_to_weigh):
    """Filter and truncate a nearest-first candidate list down to the WPs that can
    receive introduction credit for a measure.

    `candidates` is a list of `(distance, parties, year)` tuples, nearest neighbour
    first, with `parties` already run through `utils.split_parties` but not yet
    stripped of non-party authors. A candidate survives only if it has at least one
    real party author left after removing `NON_PARTY_AUTHORS`, and a known year that
    does not postdate the measure (a WP can only introduce a measure it predates).

    Stops as soon as `neighbours_to_weigh` candidates have survived: everything past
    that point in `candidates` is never even looked at, so a later candidate that
    would otherwise fail a filter never gets the chance to be excluded -- the
    iteration has already stopped.
    """
    kept = []
    for distance, parties, year in candidates:
        filtered_parties = [p for p in parties if p not in NON_PARTY_AUTHORS]
        if not filtered_parties or year is None or year > measure_year:
            continue
        kept.append((distance, filtered_parties))
        if len(kept) == neighbours_to_weigh:
            break
    return kept


def _weighted_country_credits(kept, measure_year):
    """Turn a surviving `(distance, parties)` list into `{(measure_year, party): weight}`.

    Weights are inverse-distance, normalised to sum to 1 across `kept` -- closer WPs
    get more credit. A candidate with several party authors is not split among them:
    each party on that candidate receives the candidate's full weight, so the total
    credited weight for a measure can exceed 1 once any candidate has co-authors.

    Rounded to 10 decimal places: `distance` comes from a nearest-neighbour lookup
    over embeddings, and the underlying numpy distance computation is not guaranteed
    bit-identical run-to-run. 10dp is far below any precision this figure is read at,
    and rounding here (rather than only at output time) keeps every downstream sum
    built from already-stable numbers.
    """
    weight_normaliser = sum(1 / distance for distance, _ in kept)
    credits = {}
    for distance, parties in kept:
        doc_weight = (1 / distance) / weight_normaliser
        for p in parties:
            key = (measure_year, p)
            credits[key] = credits.get(key, 0) + doc_weight
    return {key: round(value, 10) for key, value in credits.items()}


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

            resolved_candidates = []
            for uuid, distance in candidates:
                representation = document_getter.get_document_representation(uuid)
                parties = utils.split_parties(representation.get("parties", []))
                year = representation.get("year")
                resolved_candidates.append((distance, parties, year))

            kept = _select_introducing_neighbours(resolved_candidates, measure["year"], neighbours_to_weigh)

            # Inverse-distance weight across the surviving neighbours (normalised to 1).
            for key, weight in _weighted_country_credits(kept, measure["year"]).items():
                self.country_sums_by_year[key] = self.country_sums_by_year.get(key, 0) + weight

        # A (year, country) total can sum credits from several measures adopted that
        # year, so even though each credit is already rounded, ordinary binary-float
        # addition of several rounded numbers is not itself guaranteed to land on a
        # clean 10dp value. Rounding again here keeps the figure actually written out
        # tidy, on top of the run-to-run stability the per-credit rounding provides.
        self.country_sums_by_year = {key: round(value, 10) for key, value in self.country_sums_by_year.items()}

        self.country_sums = {}
        for (_, country), weight in self.country_sums_by_year.items():
            self.country_sums[country] = self.country_sums.get(country, 0) + weight
        self.country_sums = {country: round(value, 10) for country, value in self.country_sums.items()}

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