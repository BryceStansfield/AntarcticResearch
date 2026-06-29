import multiprocessing
import pandas
from embeddings.document_embeddings import *
from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations
from embeddings.working_paper_censorship import get_working_paper_paths, censor_text, COUNTRIES

CENSORED_WORKING_PAPER_TYPE = "CensoredWorkingPaperV1"

def _embed_with_retry(*args):
    for attempt in range(3):
        try:
            return get_or_generate_embedding(*args)
        except Exception:
            if attempt == 2:
                raise

def embed_document_set(to_embed):
    with multiprocessing.Pool(processes=200) as pool:
        pool.starmap(_embed_with_retry, to_embed)

def embed_all_measures():
    pd = pandas.read_csv("data/MeasureCorpusEnriched.csv")

    to_embed = []

    for row in pd.itertuples():
        if pandas.isna(row.Content):
            continue
        
        doc_num = row.Document_Number
        doc_id = measure_id_to_uuid(doc_num)
        text_rep = get_representation_of_measure(row)

        if not has_embedding(doc_id):
            to_embed.append((doc_id, "measure", text_rep,))
    
    embed_document_set(to_embed)

def embed_all_censored_working_papers(countries=COUNTRIES):
    to_embed = []
    print("Hashing censored working papers for embedding")
    for path in get_working_paper_paths():
        censored_text = censor_text(path.read_text(encoding="utf-8", errors="ignore"), countries)
        to_embed.extend(get_wp_ip_embedding_args(censored_text, CENSORED_WORKING_PAPER_TYPE))

    print("Embedding censored working papers")
    embed_document_set(to_embed)

def embed_all():
    print("Embedding Measures")
    embed_all_measures()
    print("Done Embedding Measures")

    ip_wp_file_paths = map_all_wp_ip_file_locations()
    ip_wp_to_embed = []
    print("Hashing ips and wps for embedding")
    for path in ip_wp_file_paths.values():
        if "/wp/" in path:
            t = "WorkingPaper"
        elif "/ip/" in path:
            t = "InformationPaper"

        with open(path, "r") as f:
            ip_wp_to_embed.extend(get_wp_ip_embedding_args(f.read(), t))

    print("Embedding ips and wps")
    embed_document_set(ip_wp_to_embed)

    print("Embedding censored working papers")
    embed_all_censored_working_papers()

if __name__ == "__main__":
    embed_all()