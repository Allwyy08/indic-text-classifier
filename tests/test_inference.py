"""
Unit tests for Tamil text classification inference engine and edge-case handling.
"""

import os
import pytest
from src.inference import TamilClassifierPredictor, get_predictor


def test_predictor_missing_model_handling():
    # Should not crash on nonexistent model path
    dummy_predictor = TamilClassifierPredictor(model_dir="models/nonexistent_task", task_name="test_dummy")
    assert dummy_predictor.is_loaded is False

    res = dummy_predictor.predict("இது ஒரு சோதனை வாக்கியம்.")
    assert res["success"] is False
    assert "not loaded" in res["error"] or "not found" in res["error"]
    assert res["predicted_label"] is None
    assert res["confidence_pct"] == 0.0


def test_remote_repo_failure_handling():
    """Verify that a nonexistent or failing remote model repo returns a clear, descriptive error."""
    predictor = TamilClassifierPredictor(
        model_dir="models/nonexistent_task",
        task_name="test_remote",
        remote_repo="ai4bharat/nonexistent-failing-test-repo-12345"
    )
    assert predictor.is_loaded is False
    assert predictor.load_error is not None
    assert "Failed to download model weights from remote repository" in predictor.load_error

    res = predictor.predict("இது ஒரு சோதனை வாக்கியம்.")
    assert res["success"] is False
    assert "Failed to download model weights from remote repository" in res["error"]


def test_remote_model_repositories_configured():
    """Verify that confirmed Hugging Face model repository IDs are properly configured in config.yaml."""
    import yaml
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["tasks"]["sentiment"]["remote_model_repo"] == "Allwynn07/indicbert-tamil-sentiment"
    assert cfg["tasks"]["news_authenticity"]["remote_model_repo"] == "Allwynn07/indicbert-tamil-news-authenticity"


def test_predictor_empty_and_short_input():
    predictor = get_predictor("sentiment", "models/sentiment")
    
    # Empty string
    res_empty = predictor.predict("")
    assert res_empty["success"] is False
    assert "too short or empty" in res_empty["error"]

    # Only whitespace
    res_spaces = predictor.predict("    ")
    assert res_spaces["success"] is False

    # Under minimum character length
    res_short = predictor.predict("அ")
    assert res_short["success"] is False


def test_predictor_output_contract_format():
    # Verify the structure and return types
    predictor = get_predictor("sentiment", "models/sentiment")
    if predictor.is_loaded:
        res = predictor.predict("இந்த திரைப்படம் மிகவும் அருமையாக உள்ளது.")
        assert res["success"] is True
        assert isinstance(res["predicted_label"], str)
        assert 0.0 <= res["confidence_pct"] <= 100.0
        assert isinstance(res["probabilities"], dict)
        assert len(res["probabilities"]) == 3
        # Sum of probabilities should be close to 100%
        prob_sum = sum(res["probabilities"].values())
        assert 98.0 <= prob_sum <= 102.0
        assert res["latency_ms"] > 0


def test_sentiment_label_mapping_contract():
    """Verify that model config and config.yaml have identical id2label and label2id mappings."""
    import json
    import yaml

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sent_cfg = cfg["tasks"]["sentiment"]
    assert sent_cfg["id2label"] == {0: "Negative", 1: "Neutral", 2: "Positive"}
    assert sent_cfg["label2id"] == {"Negative": 0, "Neutral": 1, "Positive": 2}

    model_cfg_path = "models/sentiment/config.json"
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            m_cfg = json.load(f)
        id2label = {int(k): v for k, v in m_cfg.get("id2label", {}).items()}
        assert id2label == {0: "Negative", 1: "Neutral", 2: "Positive"}
        assert m_cfg.get("label2id") == {"Negative": 0, "Neutral": 1, "Positive": 2}


def test_sentiment_positive_sample_prediction():
    """Verify that the intended Positive sample genuinely predicts Positive with high confidence."""
    predictor = get_predictor("sentiment", "models/sentiment")
    if predictor.is_loaded:
        pos_text = "இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!"
        res = predictor.predict(pos_text)
        assert res["success"] is True
        assert res["predicted_label"] == "Positive", f"Expected Positive, got {res['predicted_label']} with probs {res['probabilities']}"
        assert res["confidence_pct"] > 50.0


def test_sentiment_all_classes_prediction():
    """Verify that Positive, Negative, and Neutral sample statements are distinguished correctly."""
    predictor = get_predictor("sentiment", "models/sentiment")
    if predictor.is_loaded:
        samples = [
            ("இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!", "Positive"),
            ("இந்த சேவை மிகவும் மோசமாக உள்ளது, எனக்கு மிகவும் ஏமாற்றமாக இருக்கிறது.", "Negative"),
            ("நான் ஒரு ஆசிரியர்.", "Neutral")
        ]
        for text, expected in samples:
            res = predictor.predict(text)
            assert res["success"] is True
            assert res["predicted_label"] == expected, f"Expected {expected} for '{text}', got {res['predicted_label']}"
            assert res["confidence_pct"] > 50.0


def test_direct_vs_ui_inference_parity():
    """Verify strict mathematical parity between direct PyTorch forward pass and UI Predictor."""
    import torch

    model_dir = "models/sentiment"
    if os.path.exists(os.path.join(model_dir, "config.json")):
        predictor = get_predictor("sentiment", model_dir)
        if predictor.is_loaded:
            tokenizer = predictor.tokenizer
            model = predictor.model
            model.eval()

            text = "இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!"

            enc = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                enc_dev = {k: v.to(predictor.device) for k, v in enc.items()}
                out = model(**enc_dev)
                direct_logits = out.logits[0].cpu()
                direct_probs = torch.softmax(direct_logits, dim=-1)

            pred_idx = int(torch.argmax(direct_logits).item())
            direct_label = model.config.id2label[pred_idx] if pred_idx in model.config.id2label else model.config.id2label[str(pred_idx)]

            ui_res = predictor.predict(text)
            assert ui_res["predicted_label"] == direct_label
            assert abs(ui_res["confidence_pct"] - (direct_probs[pred_idx].item() * 100)) < 0.1


def test_checkpoint_and_evaluation_artifacts_consistency():
    """Verify consistency between saved model metadata and output evaluation metrics."""
    import json

    meta_path = "models/sentiment/training_metadata.json"
    metrics_path = "outputs/sentiment/metrics.json"

    if os.path.exists(meta_path) and os.path.exists(metrics_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Check that both reflect evaluation on the test split
        assert "test_metrics" in meta or "accuracy" in meta.get("validation_metrics", {})
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.50
        assert metrics["macro_f1"] > 0.50

    import gc
    from src.inference import _PREDICTOR_REGISTRY
    _PREDICTOR_REGISTRY.clear()
    gc.collect()


