# Dataset Documentation

This directory contains the dataset preparation artifacts for the two Tamil NLP classification tasks.

## Tasks & Dataset Sources

### 1. Task A: Tamil Sentiment Analysis
- **Benchmark Source**: DravidianLangTech / NAACL 2021 Tamil Sentiment Analysis Benchmark + IndicGLUE validation text.
- **Classes**:
  - `Positive`
  - `Negative`
  - `Neutral` (mapped from unpolarized / neutral state annotations)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Research Open Access.
- **Language**: Tamil (`ta`).

### 2. Task B: Tamil News Authenticity Classification
- **Benchmark Source**: Springer SPELLL 2022 peer-reviewed publication:
  *"A Novel Dataset for Fake News Detection in Tamil Regional Language"* (AAnirudh07 / SPELLL 2022, Springer CCIS).
- **Classes**:
  - `Real News` (Authenticity = 0, verified authentic journalism)
  - `Fake News` (Authenticity = 1, verified misinformation / rumors)
- **License**: MIT License.
- **Language**: Tamil (`ta`) native script.

## Directory Structure
- `data/sentiment/`: Train, validation, and test CSVs for Task A.
- `data/news_authenticity/`: Train, validation, and test CSVs for Task B.

Run `python prepare_data.py` to download, validate, preprocess, stratify, and save both datasets locally.

