"""Prepare working-paper authorship data for LLM fine-tuning.

The model expects the target variable of the dataset to be named labels.
labels need to be binary vectors of length #labels, indicating which labels are true/false for a given sample, i.e., a multi-hot label encoding.
When using PyTorch in the backend, the labels vectors need to be floating-point numbers, not integers. This is because AutoModelForSequenceClassification uses BCEWithLogitsLoss and no automatic type casting takes place.

Emits one JSONL file per split (train / validation / test) for every (granularity, censorship
method), laid out so they load straight into the 🤗 ``datasets`` library:

    from datasets import load_dataset
    ds = load_dataset("json", data_files={
        "train":      "data/finetuning/full/raw/train.jsonl",
        "validation": "data/finetuning/full/raw/validation.jsonl",
        "test":       "data/finetuning/full/raw/test.jsonl",
    })

Each record is a working paper (or one sentence-chunk of it, at finer granularities) with a
multi-hot float ``labels`` vector aligned to ``COUNTRIES``, ready for
``AutoModelForSequenceClassification`` with ``problem_type="multi_label_classification"``
(see ``fr_finetuned_model.py``).

The document-level split is the SAME deterministic split the embedding benchmark uses
(``split_documents``), and chunking happens within each split, so the held-out test papers
stay reserved and there is no chunk-level leakage across splits.

Note: the ``llm_censorship`` method reads the LLM phrase cache (and would trigger live calls
on a miss), so populate it first with ``python -m embeddings.working_paper_censorship``.
"""
import json
import pathlib

from sentence_splitter import chunk_sentences
from working_paper_authorship.country_authorship_classifier import (
    COUNTRIES,
    CENSORSHIP_METHODS,
    GRANULARITIES,
    granularity_label,
    load_working_papers,
    split_documents,
    _apply_censorship,
)

OUTPUT_DIR = pathlib.Path("data/finetuning")
# HuggingFace's conventional split names; map onto split_documents' train/val/test.
SPLIT_NAMES = ("train", "validation", "test")


def _granularity_chunks(text: str, granularity) -> list[str]:
    """The text units for one document at a granularity: the whole document for "full",
    otherwise groups of ``granularity`` sentences (empties dropped by chunk_sentences)."""
    if granularity == "full":
        return [text] if text.strip() else []
    return chunk_sentences(text, granularity)


def _record(rec: dict, method: str, text: str, granularity) -> dict:
    """One fine-tuning example: (chunk of) censored text + multi-hot float labels + metadata."""
    label = rec["label"]
    return {
        "stem": rec["stem"],
        "text": text,
        "labels": [float(x) for x in label],  # float multi-hot for BCEWithLogitsLoss
        "countries": [c for c, on in zip(COUNTRIES, label) if on],
        "author": rec["author"],
        "method": method,
        "granularity": granularity_label(granularity),
    }


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(methods=CENSORSHIP_METHODS, granularities=GRANULARITIES,
            output_dir: pathlib.Path = OUTPUT_DIR) -> None:
    """Write ``{output_dir}/{granularity}/{method}/{split}.jsonl`` for every combination."""
    print("Loading working papers + authors...")
    records = load_working_papers()
    train, val, test = split_documents(records)
    splits = dict(zip(SPLIT_NAMES, (train, val, test)))
    print(f"  docs: {len(records)} (train {len(train)}, validation {len(val)}, test {len(test)})")

    for method in methods:
        print(f"\nCensoring with '{method}'...")
        for split_name, recs in splits.items():
            # Censor each document once for this method, then re-chunk at every granularity.
            censored = [(rec, _apply_censorship(rec, method)) for rec in recs]
            counts = []
            for granularity in granularities:
                rows = [
                    _record(rec, method, chunk, granularity)
                    for rec, text in censored
                    for chunk in _granularity_chunks(text, granularity)
                ]
                path = output_dir / granularity_label(granularity) / method / f"{split_name}.jsonl"
                _write_jsonl(path, rows)
                counts.append(f"{granularity_label(granularity)} {len(rows)}")
            print(f"  {split_name}: {len(recs)} docs -> " + ", ".join(counts))

    print(f"\nDone. Label order (index -> country): {dict(enumerate(COUNTRIES))}")
    print("Load with load_dataset_dict(granularity, method) or datasets.load_dataset('json', ...).")


def load_dataset_dict(granularity="full", method: str = "raw", output_dir: pathlib.Path = OUTPUT_DIR):
    """Convenience loader returning a 🤗 ``DatasetDict`` for one (granularity, method)'s splits."""
    import datasets
    base = output_dir / granularity_label(granularity) / method
    return datasets.load_dataset("json", data_files={
        split: str(base / f"{split}.jsonl") for split in SPLIT_NAMES
    })


if __name__ == "__main__":
    prepare()
