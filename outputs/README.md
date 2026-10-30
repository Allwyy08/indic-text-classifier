# Evaluation Outputs & Artifacts

This directory stores evaluation metrics, classification reports, confusion matrix figures, and training logs for both tasks.

## Structure
- `outputs/sentiment/`:
  - `metrics.json`: Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall, validation loss.
  - `classification_report.json`: Per-class precision, recall, and f1-score.
  - `confusion_matrix.png`: Normalized confusion matrix visualization.
  - `baseline_comparison.json`: IndicBERT vs. TF-IDF + Logistic Regression benchmark comparison.
- `outputs/news_authenticity/`:
  - `metrics.json`: Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall, validation loss.
  - `classification_report.json`: Per-class precision, recall, and f1-score.
  - `confusion_matrix.png`: Normalized confusion matrix visualization.
  - `baseline_comparison.json`: IndicBERT vs. TF-IDF + Logistic Regression benchmark comparison.

