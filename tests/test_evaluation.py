"""
Unit tests for the evaluation pipeline and baseline comparison.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.train_utils import compute_metrics
from src.evaluation import train_and_evaluate_baseline, plot_and_save_confusion_matrix


def test_compute_metrics():
    # Synthetic logits and labels
    logits = np.array([
        [2.5, 0.1, -1.0],  # Pred: 0
        [-0.5, 3.2, 0.2],  # Pred: 1
        [0.0, 0.5, 2.1],   # Pred: 2
        [1.8, 0.2, 0.1]    # Pred: 0
    ])
    labels = np.array([0, 1, 2, 0])

    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_train_and_evaluate_baseline():
    train_df = pd.DataFrame({
        "text": [
            "இந்த படம் அருமை அருமை",
            "மிகவும் நல்ல திரைப்படம்",
            "மோசமான படம் குப்பை",
            "மிகவும் மோசம் பிடிக்கவில்லை"
        ],
        "label": ["Positive", "Positive", "Negative", "Negative"]
    })
    test_df = pd.DataFrame({
        "text": [
            "அருமை திரைப்படம்",
            "மோசமான குப்பை"
        ],
        "label": ["Positive", "Negative"]
    })
    id2label = {0: "Negative", 1: "Positive"}

    res = train_and_evaluate_baseline(train_df, test_df, id2label)
    assert "metrics" in res
    assert "accuracy" in res["metrics"]
    assert 0.0 <= res["metrics"]["accuracy"] <= 1.0


def test_confusion_matrix_plotting(tmp_path):
    cm_data = [[10, 2], [1, 15]]
    class_names = ["Real", "Fake"]
    plot_file = os.path.join(tmp_path, "test_cm.png")

    plot_and_save_confusion_matrix(cm_data, class_names, plot_file, title="Test CM")
    assert os.path.exists(plot_file)
    assert os.path.getsize(plot_file) > 1000

