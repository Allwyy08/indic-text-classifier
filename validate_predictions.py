"""
Reproducible validation script for testing fine-tuned IndicBERT models
across diverse Tamil text inputs, edge cases, and known ground-truth benchmark samples.

Outputs structured audit results including latency, probabilities, and accuracy.
"""

import os
import sys
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

import torch
import pandas as pd

from src.inference import get_predictor
from src.data_utils import load_splits

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_sentiment_validation() -> List[Dict[str, Any]]:
    logger.info("=== Running Task A: Sentiment Analysis Validation ===")
    model_dir = "models/sentiment"
    predictor = get_predictor("sentiment", model_dir)

    # 1. Real examples from held-out test split
    splits = load_splits("data/sentiment")
    test_df = splits["test"]

    test_samples = []
    for label in ["Positive", "Negative", "Neutral"]:
        sub = test_df[test_df["label"] == label].head(3)
        for _, row in sub.iterrows():
            test_samples.append((row["text"], label, f"Held-out Test ({label})"))

    # 2. Curated Tamil edge cases & benchmark statements
    curated_samples = [
        # UI Quick-Fill Sample Statements
        ("இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!", "Positive", "UI Sample (Positive)"),
        ("இந்த சேவை மிகவும் மோசமாக உள்ளது, எனக்கு மிகவும் ஏமாற்றமாக இருக்கிறது.", "Negative", "UI Sample (Negative)"),
        ("நான் ஒரு ஆசிரியர்.", "Neutral", "UI Sample (Neutral)"),
        # Clearly positive Tamil
        ("அருமையான சேவை மற்றும் சிறந்த தரம், மிக்க நன்றி!", "Positive", "Clearly Positive"),
        # Clearly negative Tamil
        ("மிகவும் மோசமான சேவை, பணமும் நேரமும் வீணாகிவிட்டது.", "Negative", "Clearly Negative"),
        ("பயங்கரமான ஏமாற்றம், தரம் மிகவும் மட்டம்.", "Negative", "Clearly Negative"),
        # Neutral or factual Tamil
        ("நாளை காலை ஒன்பது மணிக்கு தொடர் வண்டி புறப்படும் என்று அறிவிக்கப்பட்டுள்ளது.", "Neutral", "Factual Announcement"),
        ("தமிழகத்தில் இன்று பல இடங்களில் மிதமான மழை பெய்தது.", "Neutral", "Factual Weather Report"),
        # Informal / Code-mixed Tamil
        ("Super movie bro semma comedy scenes", "Positive", "Informal Tamil"),
        ("Waste of time romba bore padam", "Negative", "Informal Tamil"),
        # Empty and invalid edge cases
        ("", None, "Empty String"),
        ("   ", None, "Whitespace Only"),
        ("அ", None, "Single Character (< min length)")
    ]

    all_cases = test_samples + curated_samples
    results = []

    for text, expected, category in all_cases:
        res = predictor.predict(text)
        success = res["success"]
        pred_label = res["predicted_label"]
        conf = res["confidence_pct"]
        probs = res["probabilities"]
        latency = res["latency_ms"]

        if expected is not None:
            matches = (pred_label == expected) if success else False
            match_str = "MATCH" if matches else f"MISMATCH (Expected: {expected})"
        else:
            matches = None
            match_str = "CORRECT REJECTION" if not success else "UNEXPECTED SUCCESS"

        rec = {
            "task": "Sentiment",
            "category": category,
            "input_text": text if len(text) < 60 else text[:57] + "...",
            "expected": expected,
            "predicted": pred_label,
            "confidence_pct": conf,
            "match": match_str,
            "latency_ms": latency,
            "probabilities": probs,
            "device": res.get("device", "N/A"),
            "model_checkpoint": "ai4bharat/IndicBERTv2-MLM-only"
        }
        results.append(rec)

    return results


def run_news_validation() -> List[Dict[str, Any]]:
    logger.info("=== Running Task B: News Authenticity Validation ===")
    model_dir = "models/news_authenticity"
    predictor = get_predictor("news_authenticity", model_dir)

    # 1. Real examples from held-out test split
    splits = load_splits("data/news_authenticity")
    test_df = splits["test"]

    test_samples = []
    for label in ["Real News", "Fake News"]:
        sub = test_df[test_df["label"] == label].head(3)
        for _, row in sub.iterrows():
            test_samples.append((row["text"], label, f"Held-out Test ({label})"))

    # 2. Unseen Tamil headlines & Edge cases
    curated_samples = [
        # UI Quick-Fill Sample Headlines from SPELLL 2022
        ("கர்ப்பப்பை புற்றுநோயை தடுக்கும் “செர்வாவாக்” தடுப்பூசி: இந்தியாவில் முதல் முறையாக உருவாக்கம்", "Real News", "UI Sample (Real News)"),
        ('"3,000 க்கும் மேற்பட்ட கொரோனா வைரஸ் நோயாளிகளை வேண்டுமென்றே கொன்றதற்காக" ஒரு மருத்துவரை இத்தாலி கைது செய்ததாக கூறுகிறது.', "Fake News", "UI Sample (Fake News)"),
        # Real News
        ("டைமண்ட் லீக் ஃபைனலில் பட்டத்தை வென்ற முதல் இந்தியர்: வரலாறு படைத்த நீரஜ் சோப்ரா", "Real News", "Verified Real News"),
        ("மழைக்கால முன்னெச்சரிக்கை நடவடிக்கைகளை தீவிரப்படுத்த அரசு அதிகாரிகளுக்கு முதலமைச்சர் உத்தரவிட்டுள்ளார்.", "Real News", "Verified Real News"),
        # Fake News
        ("9/11 இன் ஆண்டு விழாவில் இல்ஹான் ஓமர் பார்ட்டி கொண்டாடினார் என்று வைரல் அறிக்கை.", "Fake News", "Documented Fake Claim"),
        ("மக்கள் ஒபாமாவால் இனி மீன்பிடிக்க செல்ல முடியாது என்று பரவும் செய்தி.", "Fake News", "Documented Fake Claim"),
        # Unseen arbitrary Tamil text
        ("சென்னையில் புதிய மெட்ரோ ரயில் பாதை திறப்பு விழா அடுத்த மாதம் நடைபெற உள்ளது.", None, "Unseen Headline"),
        # Edge cases
        ("", None, "Empty String"),
        ("   ", None, "Whitespace Only")
    ]

    all_cases = test_samples + curated_samples
    results = []

    for text, expected, category in all_cases:
        res = predictor.predict(text)
        success = res["success"]
        pred_label = res["predicted_label"]
        conf = res["confidence_pct"]
        probs = res["probabilities"]
        latency = res["latency_ms"]

        if expected is not None:
            matches = (pred_label == expected) if success else False
            match_str = "MATCH" if matches else f"MISMATCH (Expected: {expected})"
        else:
            matches = None
            match_str = "N/A (Unseen Claim)" if success else "CORRECT REJECTION"

        rec = {
            "task": "News Authenticity",
            "category": category,
            "input_text": text if len(text) < 60 else text[:57] + "...",
            "expected": expected,
            "predicted": pred_label,
            "confidence_pct": conf,
            "match": match_str,
            "latency_ms": latency,
            "probabilities": probs,
            "device": res.get("device", "N/A"),
            "model_checkpoint": "ai4bharat/IndicBERTv2-MLM-only"
        }
        results.append(rec)

    return results


def print_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 105)
    print(f"{'Category':<22} | {'Input Text':<35} | {'Predicted':<12} | {'Conf%':<6} | {'Status':<18} | {'Latency':<7}")
    print("=" * 105)
    for r in results:
        inp = r['input_text'].replace("\n", " ")
        pred = str(r['predicted']) if r['predicted'] else "Rejected"
        conf_str = f"{r['confidence_pct']:.1f}%" if r['confidence_pct'] else "--"
        lat_str = f"{r['latency_ms']:.1f}ms" if r['latency_ms'] else "--"
        print(f"{r['category']:<22} | {inp:<35} | {pred:<12} | {conf_str:<6} | {r['match']:<18} | {lat_str:<7}")
    print("=" * 105)


def main():
    print("=" * 70)
    print("RESEARCH-GRADE TAMIL CLASSIFICATION VALIDATION")
    print("=" * 70)

    sent_res = run_sentiment_validation()
    print_table(sent_res)

    news_res = run_news_validation()
    print_table(news_res)

    # Save to JSON
    all_res = {"sentiment": sent_res, "news_authenticity": news_res}
    with open("outputs/validation_report.json", "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    logger.info("Saved validation report to outputs/validation_report.json")


if __name__ == "__main__":
    main()

