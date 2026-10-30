"""
Command-line evaluation entrypoint for IndicBERT and Baseline models.

Usage:
  python evaluate.py --task all
  python evaluate.py --task sentiment
  python evaluate.py --task news_authenticity
"""

import os
import sys
import json
import yaml
import argparse
import logging
import pandas as pd

from src.model_utils import load_model_for_inference, get_device
from src.data_utils import load_splits
from src.evaluation import (
    evaluate_model_on_split,
    train_and_evaluate_baseline,
    plot_and_save_confusion_matrix,
    plot_and_save_comparison_chart
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned IndicBERT Models")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["sentiment", "news_authenticity", "all"],
        help="Target task to evaluate: sentiment, news_authenticity, or all"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()


def evaluate_task(task_key: str, config: dict):
    task_cfg = config["tasks"][task_key]
    model_dir = task_cfg["model_save_dir"]
    data_dir = task_cfg["data_dir"]
    output_dir = task_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{'='*70}\nEVALUATING TASK: {task_cfg['display_title']}\n{'='*70}")

    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(f"Model for task '{task_key}' not found at {model_dir}. Run training first.")

    splits = load_splits(data_dir)
    train_df = splits["train"]
    test_df = splits["test"]

    device = get_device("auto")
    model, tokenizer, id2label = load_model_for_inference(model_dir, device=device)

    # 1. Evaluate fine-tuned IndicBERT model
    logger.info("Running evaluation on test split...")
    eval_results = evaluate_model_on_split(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        id2label=id2label,
        device=device
    )

    indic_metrics = eval_results["metrics"]
    clf_report = eval_results["classification_report"]
    cm = eval_results["confusion_matrix"]
    class_names = eval_results["class_names"]

    # 2. Evaluate TF-IDF baseline
    logger.info("Training and evaluating TF-IDF + Logistic Regression baseline...")
    base_results = train_and_evaluate_baseline(
        train_df=train_df,
        test_df=test_df,
        id2label=id2label
    )
    base_metrics = base_results["metrics"]

    # 3. Save JSON artifacts
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(indic_metrics, f, indent=2)

    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(clf_report, f, indent=2)

    baseline_path = os.path.join(output_dir, "baseline_comparison.json")
    comparison_data = {
        "task": task_key,
        "indicbert_metrics": indic_metrics,
        "baseline_metrics": base_metrics,
        "baseline_classification_report": base_results["classification_report"]
    }
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    # 4. Save Plots
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(
        cm_data=cm,
        class_names=class_names,
        output_path=cm_path,
        title=f"Confusion Matrix: {task_cfg['display_title']}"
    )

    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plot_and_save_comparison_chart(
        indicbert_metrics=indic_metrics,
        baseline_metrics=base_metrics,
        output_path=chart_path,
        title=f"{task_cfg['display_title']} - Model Benchmark Comparison"
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {task_cfg['display_title']}")
    print("=" * 60)
    print("IndicBERT (Transfer Learning):")
    for k, v in indic_metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v * 100:.2f}%" if "acc" in k or "f1" in k or "prec" in k or "rec" in k else f"  {k:18s}: {v}")
    print("-" * 60)
    print("TF-IDF Baseline:")
    for k, v in base_metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v * 100:.2f}%")
    print("=" * 60)
    return comparison_data


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tasks_to_eval = ["sentiment", "news_authenticity"] if args.task == "all" else [args.task]
    for task_key in tasks_to_eval:
        evaluate_task(task_key, config)


if __name__ == "__main__":
    main()

