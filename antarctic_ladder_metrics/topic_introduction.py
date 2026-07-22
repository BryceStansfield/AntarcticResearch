"""BERTopic over OCR'd documents using OpenRouter qwen/qwen3-embedding-8b embeddings."""
from bertopic import BERTopic
from umap import UMAP

import embeddings.document_embeddings as document_embeddings
from embeddings.bertopic_backend import OpenRouterBackend, topic_vectorizer
from antarctic_ladder_metrics.constants import DECADE_BUCKETS, START_YEAR, END_YEAR
import country_meta_info

import pandas as pd

from utils import split_parties

class WPBertTopic():
    def __init__(self):
        self.document_text_getter = document_embeddings.DocumentTextGetter()
        documents = self.document_text_getter.get_all_of_type("WorkingPaper")
        self.documents = list(filter(lambda d: d["paper_language"].lower() == "english", documents))

        umap_model = UMAP(random_state=42)
        # OpenRouterBackend subclasses BERTopic's BaseEmbedder. That matters: a
        # duck-typed embedder (just an `.encode` method) is not recognised by
        # bertopic.backend.select_backend, which silently falls through to
        # all-MiniLM-L6-v2 -- so the model would run on 384-dim MiniLM with a
        # 256-token input limit instead of these Qwen3-8B embeddings.
        #
        # Labels come from c-TF-IDF. Embedding-based representation (MMR /
        # KeyBERTInspired) was tried and rejected: it ranks candidate words by
        # similarity to the topic centroid, which in an all-Antarctic corpus
        # promotes corpus-wide vocabulary ("antarctic" reached the top-10 of 112
        # of 164 topics in the combined WP+measure model, versus 4 under
        # c-TF-IDF). It also only ever affected the topic_test.txt report --
        # TopicIntroduction and TopicDiversity read the HDBSCAN assignments,
        # which representation models never touch -- while costing thousands of
        # per-word embedding calls.
        self.topic_model = BERTopic(embedding_model=OpenRouterBackend(), umap_model=umap_model, min_topic_size=5, vectorizer_model=topic_vectorizer(), verbose=True)
        self.topics, self.probs = self.topic_model.fit_transform([d["text"] for d in self.documents])

_WPBertTopicInstance = None
def get_wp_bertopic():
    global _WPBertTopicInstance

    if _WPBertTopicInstance is None:
        _WPBertTopicInstance = WPBertTopic()
    return _WPBertTopicInstance

class TopicIntroduction():
    def __init__(self):
        wp_bertopic = get_wp_bertopic()

        topic_info = wp_bertopic.topic_model.get_topic_info()

        # Writing a text report on the topics, for sanity checking
        with open("data/topic_test.txt", "w") as f:
            f.write(topic_info.to_csv(index=False))
            f.write("\n")
            for topic_id in sorted(topic_info["Topic"]):
                words = wp_bertopic.topic_model.get_topic(topic_id)
                if isinstance(words, list):
                    word_str = ", ".join(f"{w}({s:.3f})" for w, s in words)
                    f.write(f"Topic {topic_id}: {word_str}\n")
        
        # Finally, let's figure out which document is the earliest for each topic.
        topic_to_docs = {}

        for i, t in enumerate(wp_bertopic.topics):
            if t == -1:
                continue # Outlier topic

            if t in topic_to_docs:
                topic_to_docs[t].append(wp_bertopic.documents[i])
            else:
                topic_to_docs[t] = [wp_bertopic.documents[i]]
        
        earliest_docs = [min(docs, key=lambda d:d["sort_string"]) for docs in topic_to_docs.values()]
        self.yearly_topic_introduction_count = {}
        for d in earliest_docs:
            y = d["year"]
            for party in d["parties"]:
                party = country_meta_info.normalize_country_name(party)
                self.yearly_topic_introduction_count[(y, party)] = self.yearly_topic_introduction_count.get((y, party), 0) + 1
        
        self.topic_introduction_count = {}
        for k in self.yearly_topic_introduction_count:
            self.topic_introduction_count[k[1]] = self.topic_introduction_count.get(k[1], 0) + self.yearly_topic_introduction_count[k]
    
    def country_dict(self) -> dict:
        return self.topic_introduction_count

    def figure_title(self) -> str:
        return "Working Paper Idea Introduction"

    def save_full_figures(self, path: str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.yearly_topic_introduction_count.items()]
        pd.DataFrame(yearly_figures).to_csv(path)

class TopicDiversity():
    def __init__(self):
        wp_bertopic = get_wp_bertopic()

        # (country, global_topic_id, year) triples. The topic model is fit exactly
        # once, over the whole corpus, and these are its assignments -- windowing
        # below only ever filters these rows, so topic ids mean the same thing in
        # every period and are comparable across them.
        #
        # Diversity is a distinct count, not a sum, so a window's value cannot be
        # derived by partitioning the overall one: a country working one topic in
        # two decades contributes 1 overall but 1 to each decade.
        self._country_topic_years = []

        for i, t in enumerate(wp_bertopic.topics):
            if t == -1:
                continue # Outlier topic

            document = wp_bertopic.documents[i]

            # The BERTopic corpus is filtered by language only and spans 1961-2025,
            # so clip to the ladder window here. Note this clips only *this metric's
            # counting* -- the topic model itself still sees every document, so the
            # topic ids stay identical to the ones TopicIntroduction reports against.
            if document["year"] < START_YEAR or document["year"] > END_YEAR:
                continue

            countries = [country_meta_info.normalize_country_name(p) for p in split_parties(document["parties"])]

            for country in countries:
                self._country_topic_years.append((country, t, document["year"]))

        self.countries_to_topics = self._diversity_within()

    def _diversity_within(self, min_year: int | None = None, max_year: int | None = None) -> dict:
        """Distinct topics per country, optionally restricted to a closed year range."""
        countries_to_topics = {}
        for country, topic, year in self._country_topic_years:
            if min_year is not None and year < min_year:
                continue
            if max_year is not None and year > max_year:
                continue
            if country not in countries_to_topics:
                countries_to_topics[country] = set()
            countries_to_topics[country].add(topic)

        return {c: len(topics) for c, topics in countries_to_topics.items()}

    def country_dict(self) -> dict:
        return self.countries_to_topics

    def figure_title(self) -> str:
        return "Working Paper Topic Diversity"

    def save_full_figures(self, path: str):
        # Per DECADE_BUCKETS. Only the distinct-topic *count* is re-evaluated per
        # window -- the topic model and its assignments are global and untouched.
        # The buckets tile START_YEAR..END_YEAR exactly, so every document counted
        # in country_dict() falls in exactly one period; the decade values still do
        # NOT add up to it, since a topic a country worked in several decades counts
        # once overall but once per decade.
        period_figures = []
        for label, min_year, max_year in DECADE_BUCKETS:
            for country, diversity in sorted(self._diversity_within(min_year, max_year).items()):
                period_figures.append({"period": label, "country": country, "value": diversity})
        pd.DataFrame(period_figures).to_csv(path)

if __name__ == "__main__":
    print(TopicDiversity().country_dict())