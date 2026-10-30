"""
Data processing utilities for Tamil text classification.
Supports Tamil Unicode preservation, label mapping, cleaning, stratified splitting,
resampling training data only, and data leakage validation.
"""

import os
import re
import json
import logging
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_tamil_text(text: Any) -> str:
    """
    Cleans and normalizes Tamil text while preserving Tamil Unicode characters.
    - Strips leading and trailing whitespaces.
    - Normalizes consecutive whitespace, tabs, and newlines.
    - Removes unprintable control characters.
    - Preserves Tamil Unicode range (U+0B80 - U+0BFF), standard numerals, and punctuation.
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text)
    # Remove null characters and standard non-printable control chars (except spaces)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Normalize multiple whitespace characters to a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_text(text: str, min_chars: int = 3) -> bool:
    """Checks whether text has meaningful content above a minimum character length."""
    return bool(text and len(text.strip()) >= min_chars)


def check_data_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "text") -> Dict[str, int]:
    """
    Verifies that no text overlap exists between train, validation, and test splits.
    Returns the count of overlapping samples across splits.
    """
    train_texts = set(train_df[text_col].dropna().tolist())
    val_texts = set(val_df[text_col].dropna().tolist())
    test_texts = set(test_df[text_col].dropna().tolist())

    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    val_test_overlap = len(val_texts.intersection(test_texts))

    leakage_stats = {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
    }
    return leakage_stats


def split_dataset(
    df: pd.DataFrame,
    label_col: str = "label",
    text_col: str = "text",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a full cleaned dataset into stratified Train, Validation, and Test sets.
    Ensures held-out test and validation sets reflect real data distribution,
    and guarantees zero data leakage across all splits.
    """
    # Remove duplicate texts within the dataset first to prevent leakage across splits
    df_clean = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    # First split: Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        stratify=df_clean[label_col],
        random_state=random_state
    )

    # Second split: Train vs Val (relative to train_val_df)
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[label_col],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Validate zero data leakage
    leakage = check_data_leakage(train_df, val_df, test_df, text_col=text_col)
    for pair, count in leakage.items():
        if count > 0:
            logger.warning(f"Data leakage detected in {pair}: {count} shared texts!")
        else:
            logger.info(f"Leakage check passed for {pair}: 0 overlapping texts.")

    return train_df, val_df, test_df


def resample_train_data(
    train_df: pd.DataFrame,
    label_col: str = "label",
    sample_limit_per_class: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Applies balancing/subsampling ONLY to the training dataset.
    Never modifies validation or test datasets.
    """
    if sample_limit_per_class is None:
        return train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    classes = train_df[label_col].unique()
    dfs = []
    for c in classes:
        sub = train_df[train_df[label_col] == c]
        n_take = min(len(sub), sample_limit_per_class)
        dfs.append(sub.sample(n=n_take, random_state=random_state))

    balanced_train = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info(f"Resampled training set to {len(balanced_train)} samples ({sample_limit_per_class}/class).")
    return balanced_train


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """Saves train, validation, and test splits into CSV files in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    summary = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": len(train_df) + len(val_df) + len(test_df),
        "train_dist": train_df["label"].value_counts().to_dict(),
        "val_dist": val_df["label"].value_counts().to_dict(),
        "test_dist": test_df["label"].value_counts().to_dict()
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved splits to {output_dir}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return summary


def load_splits(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Loads train, validation, and test DataFrames from a directory."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"Missing one or more split files in {data_dir}. Run prepare_data.py first.")

    return {
        "train": pd.read_csv(train_path, encoding="utf-8"),
        "val": pd.read_csv(val_path, encoding="utf-8"),
        "test": pd.read_csv(test_path, encoding="utf-8")
    }
