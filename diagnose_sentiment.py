"""
Diagnostic script to trace sentiment inference pipeline outside Streamlit.
Inspects raw logits, softmax probabilities, label mappings, and model weights.
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.inference import get_predictor

def run_diagnostics():
    model_path = "models/sentiment"
    print("=" * 70)
    print("SENTIMENT PIPELINE DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Checkpoint Path: {model_path}")

    # Load tokenizer and model directly
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Base Backbone: {model.config._name_or_path}")
    print(f"Number of labels: {model.num_labels}")
    print(f"Config id2label: {model.config.id2label}")
    print(f"Config label2id: {model.config.label2id}")

    test_cases = [
        ("Positive Sample", "இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!"),
        ("Negative Sample", "இந்த சேவை மிகவும் மோசமாக உள்ளது, எனக்கு மிகவும் ஏமாற்றமாக இருக்கிறது."),
        ("Neutral Sample", "நான் ஒரு ஆசிரியர்.")
    ]

    ui_predictor = get_predictor("sentiment", model_path)

    for name, text in test_cases:
        print("\n" + "-" * 60)
        print(f"Case: {name}")
        print(f"Text: '{text}'")

        # 1. Direct PyTorch forward pass
        encoded = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        pred_idx = int(torch.argmax(logits).item())
        id2label = model.config.id2label
        pred_label = id2label[pred_idx] if pred_idx in id2label else id2label[str(pred_idx)]

        class_prob_dict = {}
        for idx in range(len(probs)):
            c_name = id2label[idx] if idx in id2label else id2label[str(idx)]
            class_prob_dict[c_name] = round(float(probs[idx].item()) * 100, 2)

        print(f"Tokens: {tokenizer.tokenize(text)}")
        print(f"Raw Logits: [Negative: {logits[0].item():.4f}, Neutral: {logits[1].item():.4f}, Positive: {logits[2].item():.4f}]")
        print(f"Class Probabilities: {class_prob_dict}")
        print(f"Predicted Class Index: {pred_idx}")
        print(f"Direct Prediction: {pred_label}")

        # 2. UI Predictor pass
        ui_res = ui_predictor.predict(text)
        print(f"UI Predictor Label: {ui_res['predicted_label']}")
        print(f"UI Predictor Confidence: {ui_res['confidence_pct']:.2f}%")
        print(f"UI Predictor Probabilities: {ui_res['probabilities']}")
        print(f"Parity (Direct == UI): {pred_label == ui_res['predicted_label']}")


if __name__ == "__main__":
    run_diagnostics()

