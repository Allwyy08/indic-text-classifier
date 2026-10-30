"""
Training pipeline utilities for fine-tuning IndicBERT on Tamil text classification tasks.
Enforces memory-conscious batching, gradient accumulation, and thread management.
"""

import os
import gc
import sys
import time
import json
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

from src.model_utils import (
    get_device,
    load_tokenizer,
    create_classification_model,
    save_model_artifacts
)
from src.data_utils import load_splits

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Computes academic classification metrics: Accuracy, Macro F1, Weighted F1, Macro P & R."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r)
    }


def prepare_hf_dataset(
    df: pd.DataFrame,
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 64
) -> Dataset:
    """Converts a pandas DataFrame into a tokenized Hugging Face Dataset."""
    df_clean = df.copy()
    df_clean["label_id"] = df_clean["label"].map(label2id)

    if df_clean["label_id"].isna().any():
        unknowns = df_clean[df_clean["label_id"].isna()]["label"].unique()
        raise ValueError(f"Found unmapped labels in dataset: {unknowns}")

    hf_ds = Dataset.from_pandas(df_clean[["text", "label_id"]])

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False
        )

    tokenized_ds = hf_ds.map(tokenize_fn, batched=True, desc="Tokenizing dataset")
    tokenized_ds = tokenized_ds.rename_column("label_id", "label")
    return tokenized_ds


def run_training_pipeline(
    task_key: str,
    config: dict,
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_lr: Optional[float] = None,
    sample_train_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end fine-tuning pipeline for a classification task.
    """
    gc.collect()
    logger.info(f"========== Starting Training Pipeline: Task = {task_key} ==========")
    if smoke_test:
        logger.info("SMOKE TEST MODE ENABLED: Running on minimal data subset for 1 epoch.")

    seed = config["project"].get("seed", 42)
    set_seed(seed)

    task_cfg = config["tasks"][task_key]
    data_dir = task_cfg["data_dir"]
    model_save_dir = task_cfg["model_save_dir"]
    output_dir = task_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters
    model_name = config["model"]["pretrained_model_name"]
    max_length = config["model"].get("max_length", 64)
    learning_rate = override_lr or float(config["training"]["learning_rate"])
    batch_size = override_batch_size or 4  # Memory-safe batch size for CPU stability
    grad_accum = 2  # Effective batch size = 8
    epochs = 1 if smoke_test else (override_epochs or int(config["training"]["num_train_epochs"]))

    id2label = {int(k): str(v) for k, v in task_cfg["id2label"].items()}
    label2id = {str(k): int(v) for k, v in task_cfg["label2id"].items()}
    num_labels = int(task_cfg["num_labels"])

    # Load splits
    splits = load_splits(data_dir)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    if smoke_test:
        train_df = train_df.head(16)
        val_df = val_df.head(8)
        test_df = test_df.head(8)
    elif sample_train_limit and len(train_df) > sample_train_limit:
        train_df = train_df.sample(n=sample_train_limit, random_state=seed).reset_index(drop=True)
        logger.info(f"Subsampled training set to {sample_train_limit} samples for resource efficiency.")

    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    # Tokenizer & Model
    tokenizer = load_tokenizer(model_name)
    model = create_classification_model(
        pretrained_model_name=model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Prepare datasets
    train_dataset = prepare_hf_dataset(train_df, tokenizer, label2id, max_length=max_length)
    val_dataset = prepare_hf_dataset(val_df, tokenizer, label2id, max_length=max_length)
    test_dataset = prepare_hf_dataset(test_df, tokenizer, label2id, max_length=max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=16,
        learning_rate=learning_rate,
        weight_decay=config["training"].get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=max(1, len(train_dataset) // (batch_size * grad_accum * 4)),
        save_total_limit=1,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Execute training
    start_time = time.time()
    train_result = trainer.train()
    training_duration_sec = time.time() - start_time

    logger.info(f"Training completed in {training_duration_sec:.2f} seconds.")

    # Validation evaluation
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    logger.info(f"Validation Metrics: {val_metrics}")

    # Test evaluation
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    logger.info(f"Test Metrics: {test_metrics}")

    # Build comprehensive metadata
    device_name = "CUDA: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    metadata = {
        "task_name": task_key,
        "display_title": task_cfg["display_title"],
        "pretrained_checkpoint": model_name,
        "architecture": type(model).__name__,
        "num_labels": num_labels,
        "classes": task_cfg["classes"],
        "hyperparameters": {
            "learning_rate": learning_rate,
            "num_train_epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "max_sequence_length": max_length,
            "weight_decay": config["training"].get("weight_decay", 0.01),
            "seed": seed
        },
        "sample_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "training_stats": {
            "train_loss": float(train_result.training_loss),
            "train_runtime_sec": float(training_duration_sec),
            "device": device_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        },
        "validation_metrics": {k.replace("eval_", ""): float(v) for k, v in val_metrics.items()},
        "test_metrics": {k.replace("test_", ""): float(v) for k, v in test_metrics.items()}
    }

    # Save best model to designated production save directory
    save_model_artifacts(
        model=trainer.model,
        tokenizer=tokenizer,
        save_dir=model_save_dir,
        metadata=metadata,
        id2label=id2label,
        label2id=label2id
    )

    # Clean intermediate checkpoint dir to conserve disk
    import shutil
    try:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    except Exception:
        pass

    logger.info(f"Model and metadata successfully saved to: {model_save_dir}")
    return metadata
