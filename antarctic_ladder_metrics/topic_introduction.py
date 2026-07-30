"""BERTopic over OCR'd documents using OpenRouter qwen/qwen3-embedding-8b embeddings."""
from bertopic import BERTopic

import embeddings.document_embeddings as document_embeddings
from embeddings.bertopic_backend import OpenRouterBackend, bertopic_umap, topic_vectorizer
from antarctic_ladder_metrics.constants import DECADE_BUCKETS, START_YEAR, END_YEAR
import country_meta_info

import pandas as pd

from utils import split_parties

class WPBertTopic():
    def __init__(self):
        self.document_text_getter = document_embeddings.DocumentTextGetter()
        documents = self.document_text_getter.get_all_of_type("WorkingPaper")
        self.documents = list(filter(lambda d: d["paper_language"].lower() == "english", documents))

        # NOTE:
        # This exists because UMAP(random_state=42) has some weird default settings, like mapping to 2d
        # And using euclidean distance. This mistake lead to bad clustering and large numbers of outliers.
        umap_model = bertopic_umap()

        self.topic_model = BERTopic(embedding_model=OpenRouterBackend(), umap_model=umap_model, min_topic_size=5, vectorizer_model=topic_vectorizer(), verbose=True)
        self.topics, self.probs = self.topic_model.fit_transform([d["text"] for d in self.documents])

_WPBertTopicInstance = None
def get_wp_bertopic():
    global _WPBertTopicInstance

    if _WPBertTopicInstance is None:
        _WPBertTopicInstance = WPBertTopic()
    return _WPBertTopicInstance

def _earliest_introductions(topics: list[int], documents: list[dict]) -> dict:
    """Pure aggregation extracted from `TopicIntroduction.__init__`.

    Groups `documents[i]` by `topics[i]` (skipping outlier topic -1), picks the
    document with the minimum `sort_string` per topic as the doc that "introduced"
    that topic, then for each earliest doc, for each of its (normalized) parties,
    tallies a yearly and a total introduction count.

    Note this mirrors the original code exactly: `parties` is iterated directly
    and only normalized (no `split_parties` call), unlike `TopicDiversity` below.

    Returns a dict with keys "earliest_docs", "yearly_topic_introduction_count"
    and "topic_introduction_count".
    """
    topic_to_docs = {}

    for i, t in enumerate(topics):
        if t == -1:
            continue # Outlier topic

        if t in topic_to_docs:
            topic_to_docs[t].append(documents[i])
        else:
            topic_to_docs[t] = [documents[i]]

    earliest_docs = [min(docs, key=lambda d:d["sort_string"]) for docs in topic_to_docs.values()]
    yearly_topic_introduction_count = {}
    for d in earliest_docs:
        y = d["year"]
        for party in d["parties"]:
            party = country_meta_info.normalize_country_name(party)
            yearly_topic_introduction_count[(y, party)] = yearly_topic_introduction_count.get((y, party), 0) + 1

    topic_introduction_count = {}
    for k in yearly_topic_introduction_count:
        topic_introduction_count[k[1]] = topic_introduction_count.get(k[1], 0) + yearly_topic_introduction_count[k]

    return {
        "earliest_docs": earliest_docs,
        "yearly_topic_introduction_count": yearly_topic_introduction_count,
        "topic_introduction_count": topic_introduction_count,
    }

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
        result = _earliest_introductions(wp_bertopic.topics, wp_bertopic.documents)
        self.yearly_topic_introduction_count = result["yearly_topic_introduction_count"]
        self.topic_introduction_count = result["topic_introduction_count"]

    def country_dict(self) -> dict:
        return self.topic_introduction_count

    def figure_title(self) -> str:
        return "Working Paper Idea Introduction"

    def save_full_figures(self, path: str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.yearly_topic_introduction_count.items()]
        pd.DataFrame(yearly_figures).to_csv(path)

def _country_topic_year_triples(topics: list[int], documents: list[dict], start_year: int, end_year: int) -> list[tuple]:
    """Pure aggregation extracted from `TopicDiversity.__init__`.

    Builds (country, global_topic_id, year) triples, one per (document,
    country-on-that-document) pair, skipping outlier topic -1 and documents
    outside the closed [start_year, end_year] window. Countries are normalized
    via `country_meta_info.normalize_country_name` after being split out with
    `utils.split_parties`.
    """
    triples = []

    for i, t in enumerate(topics):
        if t == -1:
            continue # Outlier topic

        document = documents[i]

        # The BERTopic corpus is filtered by language only and spans 1961-2025,
        # so clip to the ladder window here. Note this clips only *this metric's
        # counting* -- the topic model itself still sees every document, so the
        # topic ids stay identical to the ones TopicIntroduction reports against.
        if document["year"] < start_year or document["year"] > end_year:
            continue

        countries = [country_meta_info.normalize_country_name(p) for p in split_parties(document["parties"])]

        for country in countries:
            triples.append((country, t, document["year"]))

    return triples

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
        self._country_topic_years = _country_topic_year_triples(
            wp_bertopic.topics, wp_bertopic.documents, START_YEAR, END_YEAR
        )

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
        # Per DECADE_BUCKETS
        period_figures = []
        for label, min_year, max_year in DECADE_BUCKETS:
            for country, diversity in sorted(self._diversity_within(min_year, max_year).items()):
                period_figures.append({"period": label, "country": country, "value": diversity})
        pd.DataFrame(period_figures).to_csv(path)

if __name__ == "__main__":
    print(TopicDiversity().country_dict())