"""BERTopic over OCR'd documents using OpenRouter qwen/qwen3-embedding-4b embeddings."""
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from collections import Counter

import embeddings.document_embeddings as document_embeddings


class OpenRouterEmbedder:
    """BERTopic-compatible embedder backed by document_embeddings.py + OpenRouter."""

    def __init__(self, type) -> None:
        self.type = type
    
    def encode(self, documents: list[str], show_progress_bar: bool = False) -> np.ndarray:
        vectors = []
        for i, text in enumerate(documents):
            # Stable UUID derived from content so embeddings are cached across runs.
            args = document_embeddings.get_wp_ip_embedding_args(text, "WorkingPaper")[0]
            vec = document_embeddings.get_or_generate_embedding(
                *args
            )
            vectors.append(vec)
            if show_progress_bar and (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(documents)}")
        return np.array(vectors, dtype="float32")

class TopicIntroduction():
    def __init__(self):
        self.document_text_getter = document_embeddings.DocumentTextGetter()
        documents = self.document_text_getter.get_all_of_type("WorkingPaper")
        documents = list(filter(lambda d: d["paper_language"].lower() == "english", documents))

        topic_model = BERTopic(embedding_model=OpenRouterEmbedder("WorkingPaper"), min_topic_size=5, representation_model=[MaximalMarginalRelevance(diversity=0.9, top_n_words=10), KeyBERTInspired()], verbose=True)
        topics, probs = topic_model.fit_transform([d["text"] for d in documents])

        topic_info = topic_model.get_topic_info()

        # Writing a text report on the topics, for sanity checking
        with open("data/topic_test.txt", "w") as f:
            f.write(topic_info.to_csv(index=False))
            f.write("\n")
            for topic_id in sorted(topic_info["Topic"]):
                words = topic_model.get_topic(topic_id)
                if isinstance(words, list):
                    word_str = ", ".join(f"{w}({s:.3f})" for w, s in words)
                    f.write(f"Topic {topic_id}: {word_str}\n")
        
        # Finally, let's figure out which document is the earliest for each topic.
        topic_to_docs = {}

        for i, t in enumerate(topics):
            if t == -1:
                continue # Outlier topic

            if t in topic_to_docs:
                topic_to_docs[t].append(documents[i])
            else:
                topic_to_docs[t] = [documents[i]]
        
        earliest_docs = [min(docs, key=lambda d:d["sort_string"]) for docs in topic_to_docs.values()]
        self.first_author_counts = Counter([d["parties"][0] for d in earliest_docs])
    
    def country_dict(self) -> dict:
        return self.first_author_counts

    def figure_title(self) -> str:
        return "Working Paper Idea Introduction"
    
if __name__ == "__main__":
    TopicIntroduction()