"""
Model utility functions for loading IndicBERT tokenizer, building sequence
classification architectures, saving checkpoints, and loading for inference.
Enforces OpenMP thread management for stability on Windows Intel hybrid CPUs.
"""

import os
import sys
import json
import logging
from typing import Tuple, Dict, Any, Optional

# Enforce safe CPU thread limits to prevent 0xc0000005 memory contention on hybrid architectures
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
if not torch.cuda.is_available():
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        pass

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Determines and returns the compute device (CUDA or CPU).
    Automatically checks CUDA availability and reports device specs.
    """
    if preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:  # "auto"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using CUDA device: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
    else:
        logger.info(f"Using CPU device with {torch.get_num_threads()} threads for computation.")

    return device


def load_tokenizer(pretrained_model_name: str, use_fast: bool = True) -> PreTrainedTokenizer:
    """Loads tokenizer from Hugging Face Hub or local path."""
    logger.info(f"Loading tokenizer: {pretrained_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            use_fast=use_fast,
            keep_accents=True
        )
    except Exception as e:
        logger.warning(f"Failed with keep_accents=True, falling back to standard: {e}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=use_fast)
    return tokenizer


def create_classification_model(
    pretrained_model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int]
) -> PreTrainedModel:
    """
    Initializes a sequence classification model using the verified IndicBERT backbone.
    Sets explicit label-id mappings.
    """
    logger.info(f"Loading sequence classification model from '{pretrained_model_name}' with {num_labels} labels.")
    
    clean_id2label = {int(k): str(v) for k, v in id2label.items()}
    clean_label2id = {str(k): int(v) for k, v in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        id2label=clean_id2label,
        label2id=clean_label2id
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized: Total Params = {total_params:,}, Trainable Params = {trainable_params:,}")

    return model


def save_model_artifacts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_dir: str,
    metadata: Dict[str, Any],
    id2label: Dict[int, str],
    label2id: Dict[str, int]
):
    """
    Persists model weights, tokenizer files, label mappings, and training metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving model artifacts to {save_dir}...")

    # Save model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save explicit label mappings
    mapping_path = os.path.join(save_dir, "label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "id2label": {int(k): str(v) for k, v in id2label.items()},
            "label2id": {str(k): int(v) for k, v in label2id.items()}
        }, f, indent=2, ensure_ascii=False)

    # Save training metadata
    meta_path = os.path.join(save_dir, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully saved all artifacts to {save_dir}")


def load_model_for_inference(
    model_dir: str,
    device: Optional[torch.device] = None,
    remote_repo: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[int, str]]:
    """
    Loads saved model, tokenizer, and label mappings for inference.
    Checks local directory first. If local weights are missing (e.g. on Streamlit Cloud),
    falls back to the configured remote repository on Hugging Face Hub.
    """
    if device is None:
        device = get_device("auto")

    # 1. Check local checkpoint availability
    local_config_path = os.path.join(model_dir, "config.json")
    local_weights_exist = (
        os.path.exists(os.path.join(model_dir, "model.safetensors")) or
        os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
    )

    if os.path.exists(local_config_path) and local_weights_exist:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        mapping_file = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                id2label = {int(k): str(v) for k, v in data.get("id2label", {}).items()}
        else:
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

        logger.info(f"Loaded model from local {model_dir} on {device.type}")
        return model, tokenizer, id2label

    # 2. Local weights missing; check remote repository
    clean_remote = (remote_repo or "").strip()
    is_valid_remote = clean_remote and not clean_remote.startswith("YOUR_") and not clean_remote.startswith("username/")

    if is_valid_remote:
        logger.info(f"Local weights not found in '{model_dir}'. Attempting to load from remote repository '{clean_remote}'...")
        try:
            if os.path.exists(local_config_path):
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
            else:
                tokenizer = AutoTokenizer.from_pretrained(clean_remote)

            model = AutoModelForSequenceClassification.from_pretrained(clean_remote)
            model.to(device)
            model.eval()

            mapping_file = os.path.join(model_dir, "label_mapping.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    id2label = {int(k): str(v) for k, v in data.get("id2label", {}).items()}
            else:
                id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

            logger.info(f"Loaded model from remote repository '{clean_remote}' on {device.type}")
            return model, tokenizer, id2label
        except Exception as e:
            err_msg = (
                f"Failed to download model weights from remote repository '{clean_remote}': {e}. "
                f"Please ensure the repository exists, is public or accessible with credentials, and contains valid weights."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

    # 3. Neither local weights nor remote repository configured
    missing_msg = (
        f"Model weights (model.safetensors) not found in '{model_dir}' and no remote model repository is configured. "
        f"To enable cloud inference, configure 'remote_model_repo' in configs/config.yaml or set "
        f"HF_SENTIMENT_MODEL_REPO / HF_NEWS_MODEL_REPO in environment variables or Streamlit secrets."
    )
    logger.warning(missing_msg)
    raise FileNotFoundError(missing_msg)
