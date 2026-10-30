"""
Unit and integration tests for data processing and dataset validation.
"""

import os
import pytest
import pandas as pd
from src.data_utils import clean_tamil_text, is_valid_text, check_data_leakage, load_splits


def test_clean_tamil_text():
    # Preservation of Tamil Unicode
    tamil_raw = "  வணக்கம்   உலகு!  \n"
    assert clean_tamil_text(tamil_raw) == "வணக்கம் உலகு!"

    # Unprintable characters removal
    dirty = "தமிழ்\x00\x08ஆராய்ச்சி"
    assert clean_tamil_text(dirty) == "தமிழ்ஆராய்ச்சி"

    # None and empty handling
    assert clean_tamil_text(None) == ""
    assert clean_tamil_text("") == ""


def test_is_valid_text():
    assert is_valid_text("நல்ல செய்தி") is True
    assert is_valid_text("   ") is False
    assert is_valid_text("a", min_chars=3) is False
    assert is_valid_text(None) is False


def test_check_data_leakage():
    df1 = pd.DataFrame({"text": ["வாக்கியம் 1", "வாக்கியம் 2"]})
    df2 = pd.DataFrame({"text": ["வாக்கியம் 3"]})
    df3 = pd.DataFrame({"text": ["வாக்கியம் 4"]})

    leakage = check_data_leakage(df1, df2, df3, text_col="text")
    assert leakage["train_val_overlap"] == 0
    assert leakage["train_test_overlap"] == 0
    assert leakage["val_test_overlap"] == 0

    # Test intentional overlap
    df_leak = pd.DataFrame({"text": ["வாக்கியம் 1"]})
    leakage_detected = check_data_leakage(df1, df_leak, df3, text_col="text")
    assert leakage_detected["train_val_overlap"] == 1


def test_sentiment_dataset_splits_exist_and_valid():
    splits = load_splits("data/sentiment")
    assert "train" in splits and "val" in splits and "test" in splits

    expected_labels = {"Positive", "Negative", "Neutral"}
    for split_name, df in splits.items():
        assert "text" in df.columns and "label" in df.columns
        assert len(df) > 0
        actual_labels = set(df["label"].unique())
        assert actual_labels.issubset(expected_labels), f"Unexpected labels in {split_name}: {actual_labels}"
        # Assert no empty texts
        assert df["text"].isna().sum() == 0


def test_news_authenticity_dataset_splits_exist_and_valid():
    splits = load_splits("data/news_authenticity")
    assert "train" in splits and "val" in splits and "test" in splits

    expected_labels = {"Real News", "Fake News"}
    for split_name, df in splits.items():
        assert "text" in df.columns and "label" in df.columns
        assert len(df) > 0
        actual_labels = set(df["label"].unique())
        assert actual_labels.issubset(expected_labels), f"Unexpected labels in {split_name}: {actual_labels}"
        assert df["text"].isna().sum() == 0

