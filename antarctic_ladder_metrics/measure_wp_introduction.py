import embeddings.document_embeddings

class MeasureWPIntroducers():
    def __init__(self):
        document_getter = embeddings.document_embeddings.DocumentTextGetter()
        measures = document_getter.get_all_of_type("measure")
        print(measures)

        working_paper_getter = embeddings.document_embeddings.EmbeddingLookerUpper("WorkingPaper")
        
        closest_docs = []
        for measure in measures:
            closest_docs.extend(working_paper_getter.get_nearest_neighbours(measure["uuid"], 5))
        print(closest_docs)
        quit()
        
        docs = [document_getter.get_document_representation(d) for d in closest_docs]
        doc_parties = [d["parties"] for d in docs]

        country_sums = {}
        for pl in doc_parties:
            for p in pl:
                if p in country_sums:
                    country_sums[p] += 1/len(pl)
                else:
                    country_sums[p] = 1/len(pl)
        
        print(country_sums)

if __name__ == "__main__":
    MeasureWPIntroducers()