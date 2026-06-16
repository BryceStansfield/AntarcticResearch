import embeddings.document_embeddings
import utils
from antarctic_ladder_metrics.constants import *

class MeasureWPIntroducers():
    def __init__(self, neighbours_to_weigh = 3):
        document_getter = embeddings.document_embeddings.DocumentTextGetter()
        measures = list(filter(lambda m: START_YEAR <= m["year"] and m["year"] <= END_YEAR, document_getter.get_all_of_type("measure")))
        # TODO: Flag that this is mostly ANNEX-V measures...

        working_paper_getter = embeddings.document_embeddings.EmbeddingLookerUpper("WorkingPaper")
        
        closest_docs = []
        for measure in measures:
            closest_docs.append(working_paper_getter.get_nearest_neighbours(measure["uuid"], neighbours_to_weigh))
        
        docs = []
        for doc_set in closest_docs:
            weight = sum(doc_set[i][1] for i in range(neighbours_to_weigh))
            docs.append([(d[0], (d[1])/weight) for d in doc_set[:neighbours_to_weigh]])
        
        doc_parties = [(utils.split_parties(document_getter.get_document_representation(d[0])["parties"]), d[1]) for d_set in docs for d in d_set ]

        self.country_sums = {}
        for pl in doc_parties:
            for p in pl[0]:
                if p in self.country_sums:
                    self.country_sums[p] += (1/len(pl)) * pl[1]
                else:
                    self.country_sums[p] = (1/len(pl)) * pl[1]

    def country_dict(self) -> dict:
        return dict(self.country_sums)

    def figure_title(self) -> str:
        return "Measure WP Introductions"

if __name__ == "__main__":
    MeasureWPIntroducers()