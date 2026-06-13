"""BERTopic over OCR'd documents using OpenRouter qwen/qwen3-embedding-4b embeddings."""

import hashlib
import pathlib

import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

import document_embeddings

DATA_DIR = pathlib.Path("data/test")

class OpenRouterEmbedder:
    """BERTopic-compatible embedder backed by document_embeddings.py + OpenRouter."""

    def encode(self, documents: list[str], show_progress_bar: bool = False) -> np.ndarray:
        vectors = []
        for i, text in enumerate(documents):
            # Stable UUID derived from content so embeddings are cached across runs.
            doc_uuid = hashlib.sha256(text.encode()).hexdigest()
            vec = document_embeddings.get_or_generate_embedding(
                doc_uuid, 0, "bertopic", text
            )
            vectors.append(vec)
            if show_progress_bar and (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(documents)}")
        return np.array(vectors, dtype="float32")


def load_docs(data_dir: pathlib.Path = DATA_DIR) -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts) for all .txt files in data_dir."""
    paths = sorted(data_dir.glob("*.txt"))
    doc_ids = [p.stem for p in paths]
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return doc_ids, texts


def main():
    doc_ids, texts = load_docs()
    print(f"Loaded {len(texts)} documents from {DATA_DIR}/")

    topic_model = BERTopic(embedding_model=OpenRouterEmbedder(), representation_model=[MaximalMarginalRelevance(diversity=0.9, top_n_words=10), KeyBERTInspired()], verbose=True)
    topics, probs = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()
    print("\nTopic info:")
    print(topic_info)

    with open("data/topic_test.txt", "w") as f:
        f.write(topic_info.to_csv(index=False))
        f.write("\n")
        for topic_id in sorted(topic_info["Topic"]):
            words = topic_model.get_topic(topic_id)
            if isinstance(words, list):
                word_str = ", ".join(f"{w}({s:.3f})" for w, s in words)
                f.write(f"Topic {topic_id}: {word_str}\n")

    return topic_model, doc_ids, topics, probs


if __name__ == "__main__":
    main()
