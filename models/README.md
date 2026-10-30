# Model Artifacts Directory

This directory stores the fine-tuned model weights, tokenizer configuration, label mappings, and training metadata.

## Structure
- `models/sentiment/`: Best fine-tuned checkpoint for Task A (Tamil Sentiment Analysis).
- `models/news_authenticity/`: Best fine-tuned checkpoint for Task B (Tamil News Authenticity Classification).

Each model folder contains:
- `config.json`: Model architecture and hyperparameter configuration.
- `model.safetensors` or `pytorch_model.bin`: Trained weights.
- `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` / tokenizer assets: Tokenizer files.
- `training_metadata.json`: Timestamp, hyperparameters, final validation metrics, and device details.
- `label_mapping.json`: Explicit id-to-label and label-to-id dictionaries.

