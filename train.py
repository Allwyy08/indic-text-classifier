"""
Command-line training entrypoint for fine-tuning IndicBERT on Tamil classification tasks.

Usage examples:
  python train.py --task sentiment --smoke-test
  python train.py --task news_authenticity --epochs 3
  python train.py --task all
"""

import argparse
import sys
import yaml
import logging

from src.train_utils import run_training_pipeline

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune IndicBERT for Tamil Text Classification")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["sentiment", "news_authenticity", "all"],
        help="Target classification task: sentiment, news_authenticity, or all"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test on minimal data to verify pipeline without full training"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit training set size for faster execution")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tasks_to_run = ["sentiment", "news_authenticity"] if args.task == "all" else [args.task]

    results = {}
    for task_key in tasks_to_run:
        logger.info(f"\n{'='*70}\nLAUNCHING TRAINING FOR: {task_key.upper()}\n{'='*70}")
        meta = run_training_pipeline(
            task_key=task_key,
            config=config,
            smoke_test=args.smoke_test,
            override_epochs=args.epochs,
            override_batch_size=args.batch_size,
            override_lr=args.lr,
            sample_train_limit=args.sample_limit
        )
        results[task_key] = meta

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY REPORT")
    print("=" * 70)
    for task_key, meta in results.items():
        print(f"Task: {meta['display_title']}")
        print(f"  Checkpoint: {meta['pretrained_checkpoint']}")
        print(f"  Architecture: {meta['architecture']}")
        print(f"  Training Runtime: {meta['training_stats']['train_runtime_sec']:.2f}s")
        print(f"  Final Training Loss: {meta['training_stats']['train_loss']:.4f}")
        print("  Validation Metrics:")
        for k, v in meta['validation_metrics'].items():
            if isinstance(v, float):
                print(f"    - {k}: {v:.4f}")
        print("  Test Metrics:")
        for k, v in meta['test_metrics'].items():
            if isinstance(v, float):
                print(f"    - {k}: {v:.4f}")
        print("-" * 70)
    print("All tasks finished successfully!\n")


if __name__ == "__main__":
    main()

