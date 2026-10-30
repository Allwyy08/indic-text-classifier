"""
Inference engine for Tamil text classification using fine-tuned IndicBERT models.
Supports batched and single-sentence prediction, confidence scores, probability
distributions, and caching to prevent reloading models on each request.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List

import torch
import numpy as np

from src.model_utils import load_model_for_inference, get_device
from src.data_utils import clean_tamil_text, is_valid_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TamilClassifierPredictor:
    """
    Cached, thread-safe inference predictor for a specific classification task.
    """

    def __init__(
        self,
        model_dir: str,
        task_name: str,
        device: Optional[torch.device] = None,
        remote_repo: Optional[str] = None
    ):
        self.model_dir = model_dir
        self.task_name = task_name
        self.device = device or get_device("auto")
        self.remote_repo = remote_repo
        self.model = None
        self.tokenizer = None
        self.id2label = {}
        self.is_loaded = False
        self.load_error = None
        self._load_model()

    def _load_model(self):
        try:
            self.model, self.tokenizer, self.id2label = load_model_for_inference(
                self.model_dir, device=self.device, remote_repo=self.remote_repo
            )
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Initialized predictor for '{self.task_name}' on {self.device.type}")
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Error loading model for '{self.task_name}': {e}")
            self.is_loaded = False

    def predict(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        """
        Runs inference on a single Tamil text string.
        Returns:
            predicted_label: str
            confidence_pct: float (0.0 to 100.0)
            probabilities: Dict[str, float] (each 0.0 to 100.0)
            latency_ms: float
            device: str
        """
        cleaned = clean_tamil_text(text)
        if not is_valid_text(cleaned):
            return {
                "success": False,
                "error": "Input text is too short or empty. Please enter meaningful Tamil text.",
                "predicted_label": None,
                "confidence_pct": 0.0,
                "probabilities": {},
                "latency_ms": 0.0
            }

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            err_msg = self.load_error or f"Model for '{self.task_name}' is not loaded. Please train or provide model artifacts."
            return {
                "success": False,
                "error": err_msg,
                "predicted_label": None,
                "confidence_pct": 0.0,
                "probabilities": {},
                "latency_ms": 0.0
            }

        start_time = time.perf_counter()

        with torch.no_grad():
            inputs = self.tokenizer(
                cleaned,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        pred_id = int(np.argmax(probs))
        pred_label = self.id2label.get(pred_id, f"Class {pred_id}")
        confidence = float(probs[pred_id]) * 100.0

        prob_dist = {
            self.id2label.get(i, f"Class {i}"): round(float(probs[i]) * 100.0, 2)
            for i in range(len(probs))
        }

        return {
            "success": True,
            "error": None,
            "predicted_label": pred_label,
            "confidence_pct": round(confidence, 2),
            "probabilities": prob_dist,
            "latency_ms": round(latency_ms, 2),
            "device": self.device.type.upper(),
            "cleaned_text": cleaned
        }


# Singleton cache of predictors
_PREDICTOR_REGISTRY: Dict[str, TamilClassifierPredictor] = {}


def get_predictor(
    task_key: str,
    model_dir: str,
    remote_repo: Optional[str] = None
) -> TamilClassifierPredictor:
    """Retrieves or instantiates a singleton predictor for the specified task."""
    if task_key not in _PREDICTOR_REGISTRY:
        _PREDICTOR_REGISTRY[task_key] = TamilClassifierPredictor(
            model_dir=model_dir, task_name=task_key, remote_repo=remote_repo
        )
    return _PREDICTOR_REGISTRY[task_key]

