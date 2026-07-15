import multiprocessing
import pandas
from embeddings.document_embeddings import *
from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations
from embeddings.working_paper_censorship import (
    get_working_paper_paths, censor_text, llm_censor_text, author_for_stem, COUNTRIES,
)
from sentence_splitter import split_sentences

CENSORED_WORKING_PAPER_TYPE = "CensoredWorkingPaperV1"
# Per-sentence embeddings of the LLM-censored working papers, over EVERY paper (not just the
# target-country subset the authorship classifier embeds). Keyed on sha256(sentence) so they line
# up with the semantic filter's per-sentence importance labels (semantic_filter.sqlite3).
LLM_CENSORED_SENTENCE_TYPE = "LLMCensoredWPSentenceV1"

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

def embed_all_llm_censored_working_paper_sentences():
    """Embed every sentence of every LLM-censored working paper, across ALL papers with known
    authorship — not just the target-country subset the authorship classifier covers.

    Mirrors the semantic filter's sentence set exactly: for each English WP with author info,
    ``llm_censor_text`` then ``split_sentences``, one embedding per non-empty sentence, keyed on
    ``sha256(sentence)``. That is the same key ``working_paper_semantic_filter`` uses for its
    IMPORTANT/FLUFF labels, so the two tables join one-to-one for the semi-supervised experiment.

    Relies on the censorship phrase cache being populated (run ``detect_all_working_paper_phrases``
    first) so ``llm_censor_text`` is a pure cache read rather than a flood of live LLM calls.
    Already-embedded sentences are skipped, so this is safe to re-run."""
    to_embed, seen = [], set()
    print("Hashing LLM-censored working-paper sentences for embedding")
    for path in get_working_paper_paths():
        author = author_for_stem(path.stem)
        if author is None:
            continue  # no authorship info — can't LLM-censor (matches the semantic filter's skip)
        censored = llm_censor_text(path.read_text(encoding="utf-8", errors="ignore"), author)
        for raw in split_sentences(censored):
            sentence = raw.strip()
            if not sentence:
                continue
            for unit in get_wp_ip_embedding_args(sentence, LLM_CENSORED_SENTENCE_TYPE):
                if unit[0] in seen:
                    continue
                seen.add(unit[0])
                if not has_embedding(unit[0]):
                    to_embed.append(unit)

    print(f"Embedding {len(to_embed)} uncached LLM-censored sentences (of {len(seen)} unique)")
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

    print("Embedding LLM-censored working-paper sentences (all papers)")
    embed_all_llm_censored_working_paper_sentences()

if __name__ == "__main__":
    embed_all()