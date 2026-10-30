"""
Academic evaluation module for IndicBERT and TF-IDF baseline models.
Computes genuine metrics (Accuracy, Macro/Weighted F1, Precision, Recall, Per-class stats),
generates confusion matrix heatmaps, and conducts benchmark comparisons.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.model_utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_on_split(
    model,
    tokenizer,
    df: pd.DataFrame,
    id2label: Dict[int, str],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Evaluates a trained PyTorch model on a split DataFrame (e.g. Test set).
    Returns metrics, per-class report, confusion matrix, predictions, and true labels.
    """
    if device is None:
        device = get_device("auto")

    model.eval()
    model.to(device)

    texts = df["text"].tolist()
    label2id = {v: k for k, v in id2label.items()}
    true_labels = [label2id[l] for l in df["label"].tolist()]

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1).tolist()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds)

    true_arr = np.array(true_labels)
    pred_arr = np.array(all_preds)

    # Core academic metrics
    acc = accuracy_score(true_arr, pred_arr)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        true_arr, pred_arr, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        true_arr, pred_arr, average="weighted", zero_division=0
    )

    class_names = [id2label[i] for i in range(len(id2label))]

    # Per-class report
    clf_report = classification_report(
        true_arr,
        pred_arr,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(true_arr, pred_arr, labels=list(range(len(id2label))))

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_test_samples": len(df)
    }

    return {
        "metrics": metrics,
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "predictions": all_preds,
        "true_labels": true_labels
    }


def train_and_evaluate_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: Dict[int, str]
) -> Dict[str, Any]:
    """
    Trains and evaluates a competitive TF-IDF + Logistic Regression baseline
    using the identical train and test splits.
    """
    label2id = {v: k for k, v in id2label.items()}
    y_train = [label2id[l] for l in train_df["label"].tolist()]
    y_test = [label2id[l] for l in test_df["label"].tolist()]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    class_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    baseline_metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }

    return {
        "metrics": baseline_metrics,
        "classification_report": report
    }


def plot_and_save_confusion_matrix(
    cm_data: List[List[int]],
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
):
    """Generates and saves a publication-quality confusion matrix heatmap."""
    cm = np.array(cm_data)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=10, labelpad=8)
    plt.ylabel("True Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {output_path}")


def plot_and_save_comparison_chart(
    indicbert_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    output_path: str,
    title: str = "Model Comparison: IndicBERT vs. TF-IDF Baseline"
):
    """Generates a side-by-side metric comparison bar chart."""
    metric_keys = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]

    indic_vals = [indicbert_metrics.get(k, 0.0) for k in metric_keys]
    base_vals = [baseline_metrics.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, indic_vals, width, label="IndicBERT (Transfer Learning)", color="#1e3a8a")
    plt.bar(x + width/2, base_vals, width, label="TF-IDF + Logistic Regression", color="#94a3b8")

    plt.xlabel("Evaluation Metric", fontweight="bold")
    plt.ylabel("Score (0.0 - 1.0)", fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved comparison chart to: {output_path}")

