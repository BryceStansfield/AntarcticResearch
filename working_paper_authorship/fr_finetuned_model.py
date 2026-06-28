"""Fine-tune Qwen as a multi-label working-paper authorship classifier.

Trains one model per granularity for the chosen ``PARAMS["method"]`` censorship dataset
from ``prepare_data_for_finetuning``. The prepared train + validation splits are merged
into the training pool (the validation split is not held out for LLM training), and a slice
of that pool is carved off for early stopping. Final metrics are reported on the reserved
``test`` split, written to ``OUTPUT_DIR/test_report_{method}.txt``. Step-cadence checkpoints are
written during training (so the best can be reloaded at the end) but pruned immediately after,
leaving only the final model under ``data/finetuning/{granularity}/{method}/checkpoints/best``.

Edit PARAMS below and run:  python -m working_paper_authorship.fr_finetuned_model
"""
import os

# Use the expandable-segments allocator so transient long-sequence batches don't OOM the
# GPU through fragmentation (must be set before torch initialises its CUDA allocator).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import shutil

import numpy as np
from datasets import DatasetDict, concatenate_datasets
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from working_paper_authorship.country_authorship_classifier import COUNTRIES, GRANULARITIES, granularity_label
from working_paper_authorship.prepare_data_for_finetuning import OUTPUT_DIR, load_dataset_dict

# --------------------------------------------------------------------------------- params
# Everything tunable lives here — edit and re-run.
PARAMS = {
    # censorship method to fine-tune on (must already exist under data/finetuning/). Every
    # granularity is trained in turn — see train(). Overridable per run via WPAUTH_METHOD
    # (e.g. for the Vast.ai launcher), defaulting to "raw".
    "method": os.environ.get("WPAUTH_METHOD", "raw"),  # "raw" | "naive" | "llm_censorship"

    # model / tokenisation
    "model_checkpoint": "Qwen/Qwen3-8B",  # dense 8B; full FT fits one H200
    "max_length": 32000,               # tokens; longer papers are truncated

    # optimisation
    "learning_rate": 1e-5,
    # "adamw_bnb_8bit" (bitsandbytes) is recommended on a single H200: it cuts optimiser
    # state from ~96 GB to ~48 GB, leaving comfortable activation headroom for full FT.
    "optim": "adamw_bnb_8bit",
    "num_train_epochs": 50,
    # batch size 1: at max_length=32000 a batch of long papers OOMs an H200, since the
    # collator pads each batch to its longest member. grad-accum keeps the effective batch
    # (1 x 16 = 16, same as the old 4 x 4) and the optimiser-step count unchanged.
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,

    # eval / checkpoint cadence (in optimisation steps; save_steps must be a multiple of
    # eval_steps because we load the best model at the end)
    "eval_steps": 50,
    "save_steps": 50,
    "logging_steps": 10,
    "save_total_limit": 3,

    # early stopping — eval slice is carved out of the TRAIN split
    "early_stop_eval_fraction": 0.1,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0,
    "metric_for_best_model": "f1_micro",  # see compute_metrics
    "prediction_threshold": 0.5,          # sigmoid cutoff for turning logits into labels

    # hardware / misc
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "seed": 42,
}

NUM_LABELS = len(COUNTRIES)
ID2LABEL = {i: c for i, c in enumerate(COUNTRIES)}
LABEL2ID = {c: i for i, c in enumerate(COUNTRIES)}


# ---------------------------------------------------------------------------- model setup

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PARAMS["model_checkpoint"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        PARAMS["model_checkpoint"],
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def checkpoint_dir(granularity) -> str:
    """Where checkpoints land — beside the dataset dumps for this (granularity, method)."""
    return str(OUTPUT_DIR / granularity_label(granularity) / PARAMS["method"] / "checkpoints")


def prune_intermediate_checkpoints(granularity) -> None:
    """Delete the step-cadence ``checkpoint-*`` dirs, keeping only the saved ``best/`` model.

    ``load_best_model_at_end`` needs the checkpoints on disk during training; once the best model
    has been reloaded and re-saved to ``best/`` they are just dead weight (and would otherwise pile
    up ~30 GB/granularity and exhaust the disk), so drop them."""
    for path in glob.glob(os.path.join(checkpoint_dir(granularity), "checkpoint-*")):
        shutil.rmtree(path, ignore_errors=True)


# ----------------------------------------------------------------------------- data + metrics

def tokenize_splits(dataset_dict, tokenizer):
    """Tokenise ``text`` and drop every column except the tokeniser outputs + ``labels``."""
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=PARAMS["max_length"])

    keep = {"labels"}
    drop = [c for c in dataset_dict["train"].column_names if c not in keep]
    return dataset_dict.map(_tok, batched=True, remove_columns=drop)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= PARAMS["prediction_threshold"]).astype(int)
    labels = labels.astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds),
    }


# ------------------------------------------------------------------------------- training

def build_training_args(granularity) -> TrainingArguments:
    return TrainingArguments(
        output_dir=checkpoint_dir(granularity),
        learning_rate=PARAMS["learning_rate"],
        num_train_epochs=PARAMS["num_train_epochs"],
        per_device_train_batch_size=PARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=PARAMS["per_device_eval_batch_size"],
        gradient_accumulation_steps=PARAMS["gradient_accumulation_steps"],
        weight_decay=PARAMS["weight_decay"],
        warmup_ratio=PARAMS["warmup_ratio"],
        max_grad_norm=PARAMS["max_grad_norm"],
        optim=PARAMS["optim"],
        eval_strategy="steps",
        eval_steps=PARAMS["eval_steps"],
        save_strategy="steps",
        save_steps=PARAMS["save_steps"],
        logging_steps=PARAMS["logging_steps"],
        save_total_limit=PARAMS["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=PARAMS["metric_for_best_model"],
        greater_is_better=True,
        bf16=PARAMS["bf16"],
        fp16=PARAMS["fp16"],
        gradient_checkpointing=PARAMS["gradient_checkpointing"],
        seed=PARAMS["seed"],
        report_to=[],
    )


def train_one(granularity, tokenizer) -> dict:
    """Fine-tune a fresh model for one granularity and return its test-set metrics.

    The prepared train + validation splits are merged into the training pool; an early-stop
    slice is carved out of that pool. Final evaluation is on the reserved test split."""
    raw = load_dataset_dict(granularity, PARAMS["method"])
    train_pool = concatenate_datasets([raw["train"], raw["validation"]])
    split = train_pool.train_test_split(
        test_size=PARAMS["early_stop_eval_fraction"], seed=PARAMS["seed"]
    )
    tokenized = tokenize_splits(
        DatasetDict({
            "train": split["train"],
            "early_stop": split["test"],
            "test": raw["test"],
        }),
        tokenizer,
    )
    print(f"train={len(tokenized['train'])} early_stop={len(tokenized['early_stop'])} "
          f"test={len(tokenized['test'])}")

    trainer = Trainer(
        model=get_model(),
        args=build_training_args(granularity),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["early_stop"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=PARAMS["early_stopping_patience"],
            early_stopping_threshold=PARAMS["early_stopping_threshold"],
        )],
    )

    trainer.train()  # load_best_model_at_end reloads the best checkpoint before we save it
    trainer.save_model(checkpoint_dir(granularity) + "/best")
    prune_intermediate_checkpoints(granularity)  # keep only best/; drop the bulky step checkpoints

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(f"Test metrics [{granularity_label(granularity)}]: {test_metrics}")
    return test_metrics


def write_test_report(results: list[tuple]) -> None:
    """Write per-granularity test metrics to OUTPUT_DIR/test_report_{method}.txt."""
    lines = [
        "LLM FINE-TUNE — TEST-SET METRICS",
        f"Model: {PARAMS['model_checkpoint']}   Method: {PARAMS['method']}",
        f"Label order: {', '.join(COUNTRIES)}",
        "",
    ]
    for granularity, metrics in results:
        lines.append(f"[{granularity_label(granularity)}]")
        for key, value in metrics.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"test_report_{PARAMS['method']}.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote test report to {report_path}")


def train():
    set_seed(PARAMS["seed"])
    tokenizer = get_tokenizer()

    results = []
    for granularity in GRANULARITIES:
        print(f"\n=== Fine-tuning granularity {granularity_label(granularity)} "
              f"(method={PARAMS['method']}) ===")
        results.append((granularity, train_one(granularity, tokenizer)))

    write_test_report(results)
    return results


if __name__ == "__main__":
    train()