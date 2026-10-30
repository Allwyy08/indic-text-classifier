"""
Streamlit Web Application for Low-Resource Indic Text Classification Using IndicBERT.
Provides real-time Tamil sentiment analysis and news authenticity classification,
academic evaluation dashboards, model specifications, and research documentation.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
import yaml
import time
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
import torch

if not torch.cuda.is_available():
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

from src.inference import get_predictor
from src.data_utils import clean_tamil_text, is_valid_text

# Page Configuration - Light Theme Academic Layout
st.set_page_config(
    page_title="Indic Text Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Academic Light Theme Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Inter', 'Noto Sans Tamil', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #172B4D;
    }

    /* Container Padding - Compact academic layout */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 1140px !important;
    }

    /* Research Header Container */
    .header-container {
        background-color: #F5F7FA;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.15rem 1.4rem;
        margin-bottom: 1.15rem;
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #172B4D;
        letter-spacing: -0.015em;
        margin: 0 0 0.25rem 0;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        margin: 0 0 0.75rem 0;
        line-height: 1.35;
    }
    .status-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .status-badge {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        color: #475569;
    }
    .status-badge code {
        color: #172B4D;
        font-weight: 600;
        background: transparent;
        padding: 0;
        font-size: 0.78rem;
    }

    /* Section Typography */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #172B4D;
        margin: 0.35rem 0 0.2rem 0;
    }
    .section-desc {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 1rem;
    }

    /* Sample Groups */
    .sample-group-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }

    /* Result Cards */
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.15rem 1.3rem;
        margin-top: 0.85rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .result-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.65rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .result-card-title {
        font-size: 0.98rem;
        font-weight: 600;
        color: #172B4D;
        margin: 0;
    }

    /* Semantic Badges */
    .badge-positive, .badge-real {
        background-color: #E6F4EA;
        color: #137333;
        border: 1px solid #CEEAD6;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-negative, .badge-fake {
        background-color: #FCE8E6;
        color: #C5221F;
        border: 1px solid #FAD2CF;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-text {
        font-size: 0.95rem;
        font-weight: 600;
        color: #172B4D;
        margin-left: 0.5rem;
    }
    .prob-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin: 0.75rem 0 0.3rem 0;
    }
    .latency-caption {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.65rem;
        margin-bottom: 0;
    }

    /* Empty State */
    .empty-state-card {
        background-color: #F5F7FA;
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .empty-state-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.25rem;
    }
    .empty-state-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        margin: 0;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #F5F7FA;
        border-left: 3px solid #356AE6;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 0.8rem 1.1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 1.5rem;
        line-height: 1.45;
    }
    .result-disclaimer {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.65rem;
        padding-top: 0.45rem;
        border-top: 1px solid #E2E8F0;
        line-height: 1.4;
    }

    /* Specification Table */
    .spec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        margin-top: 0.4rem;
        margin-bottom: 1.15rem;
    }
    .spec-table th, .spec-table td {
        padding: 0.5rem 0.7rem;
        border: 1px solid #E2E8F0;
        text-align: left;
    }
    .spec-table th {
        background-color: #F5F7FA;
        font-weight: 600;
        color: #172B4D;
        width: 38%;
    }
    .spec-table td {
        background-color: #FFFFFF;
        color: #334155;
    }

    /* Restrained Tab Header Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.45rem 0.95rem;
        font-weight: 500;
        font-size: 0.92rem;
        color: #64748B;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #356AE6 !important;
        font-weight: 600;
        border-bottom: 2px solid #356AE6;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_app_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_app_config()


def get_remote_repo(task_key: str, app_config: dict) -> Optional[str]:
    """Resolves remote model repository from Streamlit secrets, env vars, or config.yaml."""
    env_keys = {
        "sentiment": ["HF_SENTIMENT_MODEL_REPO", "SENTIMENT_MODEL_REPO"],
        "news_authenticity": ["HF_NEWS_MODEL_REPO", "NEWS_MODEL_REPO"]
    }.get(task_key, [])

    # 1. Streamlit secrets
    try:
        if hasattr(st, "secrets"):
            for k in env_keys:
                if k in st.secrets:
                    val = str(st.secrets[k]).strip()
                    if val and not val.startswith("YOUR_"):
                        return val
    except Exception:
        pass

    # 2. Environment variables
    for k in env_keys:
        if k in os.environ:
            val = os.environ[k].strip()
            if val and not val.startswith("YOUR_"):
                return val

    # 3. config.yaml
    cfg_repo = app_config.get("tasks", {}).get(task_key, {}).get("remote_model_repo", "")
    if cfg_repo and isinstance(cfg_repo, str):
        clean = cfg_repo.strip()
        if clean and not clean.startswith("YOUR_") and not clean.startswith("username/"):
            return clean

    return None


# Task directories and remote repositories
sentiment_model_dir = config["tasks"]["sentiment"]["model_save_dir"]
news_model_dir = config["tasks"]["news_authenticity"]["model_save_dir"]
sentiment_remote_repo = get_remote_repo("sentiment", config)
news_remote_repo = get_remote_repo("news_authenticity", config)

# Check Model Availability - verifies configuration and weights on disk or remote
sentiment_weights_exist = (
    os.path.exists(os.path.join(sentiment_model_dir, "model.safetensors")) or
    os.path.exists(os.path.join(sentiment_model_dir, "pytorch_model.bin"))
)
news_weights_exist = (
    os.path.exists(os.path.join(news_model_dir, "model.safetensors")) or
    os.path.exists(os.path.join(news_model_dir, "pytorch_model.bin"))
)
sentiment_available = (
    (os.path.exists(os.path.join(sentiment_model_dir, "config.json")) and sentiment_weights_exist) or
    bool(sentiment_remote_repo)
)
news_available = (
    (os.path.exists(os.path.join(news_model_dir, "config.json")) and news_weights_exist) or
    bool(news_remote_repo)
)
device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

# Page Header Section - Restrained Research Lab Styling
st.markdown(f"""
<div class="header-container">
    <h1 class="header-title">Indic Text Classifier</h1>
    <div class="header-subtitle">Low-resource Indic language classification using fine-tuned IndicBERT models.</div>
    <div class="status-badge-container">
        <span class="status-badge">Backbone: <code>{config['model']['pretrained_model_name']}</code></span>
        <span class="status-badge">Device: <code>{device_name}</code></span>
        <span class="status-badge">Task A (Sentiment): <code>{'Ready' if sentiment_available else 'Unavailable'}</code></span>
        <span class="status-badge">Task B (Authenticity): <code>{'Ready' if news_available else 'Unavailable'}</code></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs - Clean, no emojis
tab1, tab2, tab3, tab4 = st.tabs([
    "Classification",
    "Evaluation",
    "Model Specifications",
    "Research & Methodology"
])

# Initialize stable session state for text input
if "tamil_text_input" not in st.session_state:
    st.session_state["tamil_text_input"] = ""


# Callback helpers for robust Streamlit session state management
def set_sample_text(sample_text: str):
    """Callback to safely update text area session state before widget instantiation."""
    st.session_state["tamil_text_input"] = sample_text


def clear_text_input():
    """Callback to safely reset text area session state before widget instantiation."""
    st.session_state["tamil_text_input"] = ""


# Verified Sample Tamil Corpora
SAMPLE_POSITIVE = "இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!"
SAMPLE_NEGATIVE = "இந்த சேவை மிகவும் மோசமாக உள்ளது, எனக்கு மிகவும் ஏமாற்றமாக இருக்கிறது."
SAMPLE_NEUTRAL = "நான் ஒரு ஆசிரியர்."
# Verified news headlines from Springer SPELLL 2022 corpus (data/news_authenticity)
SAMPLE_REAL_NEWS = "கர்ப்பப்பை புற்றுநோயை தடுக்கும் “செர்வாவாக்” தடுப்பூசி: இந்தியாவில் முதல் முறையாக உருவாக்கம்"
SAMPLE_FAKE_NEWS = '"3,000 க்கும் மேற்பட்ட கொரோனா வைரஸ் நோயாளிகளை வேண்டுமென்றே கொன்றதற்காக" ஒரு மருத்துவரை இத்தாலி கைது செய்ததாக கூறுகிறது.'

# ---------------------------------------------------------
# TAB 1: CLASSIFICATION WORKSPACE
# ---------------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">Classification Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Dual-task text classification using fine-tuned IndicBERT transformer models.</div>', unsafe_allow_html=True)

    # Quick-Fill Examples Section
    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.markdown('<p class="sample-group-title">Task A: Tamil Sentiment Analysis</p>', unsafe_allow_html=True)
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        sub_c1.button(
            "Positive",
            on_click=set_sample_text,
            args=(SAMPLE_POSITIVE,),
            use_container_width=True,
            help="Load positive Tamil sentiment sentence"
        )
        sub_c2.button(
            "Negative",
            on_click=set_sample_text,
            args=(SAMPLE_NEGATIVE,),
            use_container_width=True,
            help="Load negative Tamil sentiment sentence"
        )
        sub_c3.button(
            "Neutral",
            on_click=set_sample_text,
            args=(SAMPLE_NEUTRAL,),
            use_container_width=True,
            help="Load neutral Tamil sentence"
        )

    with col_ex2:
        st.markdown('<p class="sample-group-title">Task B: Tamil News Authenticity</p>', unsafe_allow_html=True)
        sub_c4, sub_c5 = st.columns(2)
        sub_c4.button(
            "Real News",
            on_click=set_sample_text,
            args=(SAMPLE_REAL_NEWS,),
            use_container_width=True,
            help="Verified Real News headline from Springer SPELLL 2022 dataset"
        )
        sub_c5.button(
            "Fake News",
            on_click=set_sample_text,
            args=(SAMPLE_FAKE_NEWS,),
            use_container_width=True,
            help="Documented Fake News claim from Springer SPELLL 2022 dataset"
        )

    # Text Input Area bound directly to stable session state key
    user_text = st.text_area(
        "Tamil text or news headline",
        key="tamil_text_input",
        height=110,
        placeholder="உங்கள் தமிழ் உரை அல்லது செய்தித் தலைப்பை இங்கே உள்ளிடவும்..."
    )

    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 7])
    classify_clicked = col_btn1.button("Classify text", type="primary", use_container_width=True)
    clear_clicked = col_btn2.button("Clear", on_click=clear_text_input, use_container_width=True)

    if classify_clicked:
        cleaned_input = clean_tamil_text(user_text)
        if not is_valid_text(cleaned_input):
            st.warning("Please enter meaningful Tamil text of at least 3 characters.")
        else:
            with st.spinner("Running IndicBERT inference..."):
                pred_col1, pred_col2 = st.columns(2)

                # --- Task A: Sentiment Prediction ---
                with pred_col1:
                    st.markdown("""
                    <div class="result-card">
                        <div class="result-card-header">
                            <span class="result-card-title">Tamil Sentiment</span>
                        </div>
                    """, unsafe_allow_html=True)

                    if sentiment_available:
                        predictor_sentiment = get_predictor("sentiment", sentiment_model_dir, remote_repo=sentiment_remote_repo)
                        res_sent = predictor_sentiment.predict(cleaned_input)

                        if res_sent["success"]:
                            label = res_sent["predicted_label"]
                            conf = res_sent["confidence_pct"]

                            badge_cls = {
                                "Positive": "badge-positive",
                                "Negative": "badge-negative",
                                "Neutral": "badge-neutral"
                            }.get(label, "badge-neutral")

                            st.markdown(f"""
                            <div style="margin-bottom: 0.85rem;">
                                <span class="{badge_cls}">{label}</span>
                                <span class="confidence-text">{conf:.1f}% confidence</span>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown('<p class="prob-label">Class Probabilities</p>', unsafe_allow_html=True)
                            for c_name in ["Negative", "Neutral", "Positive"]:
                                c_prob = res_sent["probabilities"].get(c_name, 0.0)
                                st.progress(c_prob / 100.0, text=f"{c_name}: {c_prob:.1f}%")

                            st.markdown(f'<p class="latency-caption">Latency: {res_sent["latency_ms"]:.1f} ms | Device: {res_sent["device"]}</p>', unsafe_allow_html=True)
                        else:
                            st.error(res_sent["error"])
                    else:
                        st.info("Task A checkpoint is unavailable. Local weights (model.safetensors) are not present and no remote model repository is configured.")

                    st.markdown("</div>", unsafe_allow_html=True)

                # --- Task B: News Authenticity Prediction ---
                with pred_col2:
                    st.markdown("""
                    <div class="result-card">
                        <div class="result-card-header">
                            <span class="result-card-title">Tamil News Authenticity</span>
                        </div>
                    """, unsafe_allow_html=True)

                    if news_available:
                        predictor_news = get_predictor("news_authenticity", news_model_dir, remote_repo=news_remote_repo)
                        res_news = predictor_news.predict(cleaned_input)

                        if res_news["success"]:
                            label = res_news["predicted_label"]
                            conf = res_news["confidence_pct"]

                            badge_cls = "badge-real" if "Real" in label else "badge-fake"

                            st.markdown(f"""
                            <div style="margin-bottom: 0.85rem;">
                                <span class="{badge_cls}">{label}</span>
                                <span class="confidence-text">{conf:.1f}% confidence</span>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown('<p class="prob-label">Class Probabilities</p>', unsafe_allow_html=True)
                            for c_name in ["Real News", "Fake News"]:
                                c_prob = res_news["probabilities"].get(c_name, 0.0)
                                st.progress(c_prob / 100.0, text=f"{c_name}: {c_prob:.1f}%")

                            st.markdown(f'<p class="latency-caption">Latency: {res_news["latency_ms"]:.1f} ms | Device: {res_news["device"]}</p>', unsafe_allow_html=True)
                            st.markdown("""
                            <div class="result-disclaimer">
                                Statistical model trained on Springer SPELLL 2022 research corpus. Indicates linguistic pattern similarity, not independent factual verification.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(res_news["error"])
                    else:
                        st.info("Task B checkpoint is unavailable. Local weights (model.safetensors) are not present and no remote model repository is configured.")

                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state-card">
            <div class="empty-state-title">Awaiting Input</div>
            <div class="empty-state-subtitle">Select a sample sentence above or enter Tamil text to analyze, then click <strong>Classify text</strong>.</div>
        </div>
        """, unsafe_allow_html=True)

    # Research Limitations Disclaimer Notice
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Research limitations:</strong> Sentiment predictions represent learned linguistic patterns and may not capture sarcasm, context, or every Tamil dialect. News authenticity predictions are based on patterns learned from the training dataset and do not independently verify claims against live news sources or external fact-checking services. Model confidence is not a guarantee of correctness.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 2: EVALUATION DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Academic Evaluation & Benchmark Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Quantitative performance metrics measured on held-out test splits with baseline comparisons.</div>', unsafe_allow_html=True)

    task_choice = st.radio(
        "Select classification task:",
        ["Task A: Tamil Sentiment Analysis", "Task B: Tamil News Authenticity"],
        horizontal=True
    )

    selected_task_key = "sentiment" if "Sentiment" in task_choice else "news_authenticity"
    output_dir = config["tasks"][selected_task_key]["output_dir"]
    metrics_path = os.path.join(output_dir, "metrics.json")
    report_path = os.path.join(output_dir, "classification_report.json")
    baseline_path = os.path.join(output_dir, "baseline_comparison.json")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    chart_path = os.path.join(output_dir, "comparison_chart.png")

    if not os.path.exists(metrics_path):
        st.info(f"Evaluation artifacts not found for {task_choice}. Run `python evaluate.py --task {selected_task_key}` to compute and persist test metrics.")
    else:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        num_samples = metrics.get('num_test_samples', 'N/A')
        st.markdown(f"**Test Set Evaluation Summary** &nbsp;·&nbsp; <span style='color: #64748B;'>Sample Count: N = {num_samples} held-out test samples</span>", unsafe_allow_html=True)

        # KPI Metrics Row
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.2f}%")
        m_col2.metric("Macro F1", f"{metrics.get('macro_f1', 0.0) * 100:.2f}%")
        m_col3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0.0) * 100:.2f}%")
        m_col4.metric("Macro Precision", f"{metrics.get('macro_precision', 0.0) * 100:.2f}%")
        m_col5.metric("Macro Recall", f"{metrics.get('macro_recall', 0.0) * 100:.2f}%")

        # Baseline Comparison Summary if available
        if os.path.exists(baseline_path):
            with open(baseline_path, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            b_metrics = b_data.get("baseline_metrics", {})
            b_acc = b_metrics.get("accuracy", 0.0) * 100.0
            b_f1 = b_metrics.get("macro_f1", 0.0) * 100.0
            i_acc = metrics.get("accuracy", 0.0) * 100.0
            i_f1 = metrics.get("macro_f1", 0.0) * 100.0
            acc_delta = i_acc - b_acc
            f1_delta = i_f1 - b_f1

            comp_df = pd.DataFrame([
                {"Model Architecture": "IndicBERT (Fine-Tuned)", "Accuracy": f"{i_acc:.2f}%", "Macro F1": f"{i_f1:.2f}%", "Relative Transfer Gain": "Primary Model"},
                {"Model Architecture": "TF-IDF + Logistic Regression", "Accuracy": f"{b_acc:.2f}%", "Macro F1": f"{b_f1:.2f}%", "Relative Transfer Gain": f"{acc_delta:+.2f}% Acc / {f1_delta:+.2f}% F1"}
            ])
            st.markdown("**Comparative Baseline Benchmark (Identical Test Split):**")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Visualizations Row
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.markdown("**Confusion Matrix (Held-out Test Split)**")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.caption("Confusion matrix plot not available.")

        with viz_col2:
            st.markdown("**Benchmark Comparison (IndicBERT vs TF-IDF Baseline)**")
            if os.path.exists(chart_path):
                st.image(chart_path, use_container_width=True)
            else:
                st.caption("Comparison chart not available.")

        # Detailed Classification Report
        st.markdown("**Per-Class Classification Report**")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
            filtered_rows = {k: v for k, v in rep_data.items() if isinstance(v, dict)}
            rep_df = pd.DataFrame(filtered_rows).T
            st.dataframe(rep_df.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}"
            }), use_container_width=True)


# ---------------------------------------------------------
# TAB 3: MODEL SPECIFICATIONS
# ---------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Model Specifications & Architecture Metadata</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Technical parameters of the IndicBERT transformer backbone and fine-tuning environment.</div>', unsafe_allow_html=True)

    spec_task = st.selectbox(
        "Task specification:",
        ["sentiment", "news_authenticity"],
        format_func=lambda x: config["tasks"][x]["display_title"]
    )
    meta_path = os.path.join(config["tasks"][spec_task]["model_save_dir"], "training_metadata.json")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Pretrained Backbone**")
        st.markdown(f"""
        <table class="spec-table">
            <tr><th>Checkpoint Identifier</th><td><code>{config['model']['pretrained_model_name']}</code></td></tr>
            <tr><th>Base Architecture</th><td>BERT / ALBERT Multilingual Transformer</td></tr>
            <tr><th>Organization</th><td>AI4Bharat (IIT Madras / Collaborators)</td></tr>
            <tr><th>Supported Languages</th><td>22 Indic Languages (including Tamil)</td></tr>
            <tr><th>Total Parameters</th><td>~278,043,651 parameters</td></tr>
            <tr><th>Maximum Sequence Length</th><td>128 tokens</td></tr>
            <tr><th>Vocabulary Size</th><td>250,000 subword tokens</td></tr>
            <tr><th>Tokenizer Type</th><td>SentencePiece / WordPiece with language tokens</td></tr>
        </table>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("**Fine-Tuning Configuration**")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            hp = meta.get("hyperparameters", {})
            stats = meta.get("training_stats", {})
            st.markdown(f"""
            <table class="spec-table">
                <tr><th>Task Name</th><td><code>{meta.get('task_name')}</code></td></tr>
                <tr><th>Classification Classes</th><td>{', '.join(meta.get('classes', []))}</td></tr>
                <tr><th>Learning Rate</th><td><code>{hp.get('learning_rate', 'N/A')}</code></td></tr>
                <tr><th>Effective Batch Size</th><td><code>{hp.get('effective_batch_size', hp.get('batch_size', 'N/A'))}</code></td></tr>
                <tr><th>Training Epochs</th><td><code>{hp.get('num_train_epochs', 'N/A')}</code></td></tr>
                <tr><th>Weight Decay</th><td><code>{hp.get('weight_decay', 'N/A')}</code></td></tr>
                <tr><th>Random Seed</th><td><code>{hp.get('seed', 'N/A')}</code></td></tr>
                <tr><th>Training Device</th><td>{stats.get('device', 'N/A')}</td></tr>
                <tr><th>Final Training Loss</th><td><code>{stats.get('train_loss', 0.0):.4f}</code></td></tr>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.info("Training metadata not found for this task.")

    st.markdown("**Architectural Limitations**")
    st.markdown("""
    1. **Context Window Constraint**: Maximum sequence length is bounded to 128 tokens; longer sequences are truncated.
    2. **Domain Adaptation Gap**: Informal social media code-switching (Tanglish) exhibits higher linguistic variance compared to standard literary Tamil.
    3. **Misinformation Dynamics**: Real-world news authenticity involves dynamic external factual claims that statistical language models alone cannot verify without live retrieval.
    """)


# ---------------------------------------------------------
# TAB 4: RESEARCH & METHODOLOGY
# ---------------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">Academic Research & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Theoretical background, transfer learning methodology, dataset provenance, and bibliography.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Problem Statement
    Tamil is a classical Dravidian language spoken by over 80 million people worldwide. Despite its rich literary tradition and extensive digital footprint, Tamil remains categorized as a **low-resource language** in computational linguistics due to:
    - Limited availability of standardized, high-quality labeled corpora.
    - Highly agglutinative morphology with complex inflectional and derivational structures.
    - High frequency of code-switching and informal phonetic transcriptions (Tanglish) across digital platforms.

    ### 2. Research Objectives
    1. Adapt pretrained multilingual Indic representations (`IndicBERT`) to Tamil via supervised transfer learning.
    2. Investigate sentiment polarity classification across Positive, Negative, and Neutral states.
    3. Evaluate misinformation detection on authentic Tamil regional news headlines.
    4. Benchmark fine-tuned transformer representations against classical TF-IDF + Logistic Regression baselines under identical splits.

    ### 3. Transfer Learning Workflow
    ```
    Raw Tamil Corpora (NAACL DravidianLangTech / Springer SPELLL 2022)
                 │
                 ▼
    Data Cleansing, Tamil Unicode Preservation & Zero-Leakage Splitting (80/10/10)
                 │
                 ▼
    IndicBERT Tokenizer (ai4bharat/IndicBERTv2-MLM-only)
                 │
                 ▼
    Supervised Fine-Tuning (AdamW, lr=2e-5, cross-entropy loss)
                 │
                 ▼
    Best Checkpoint Selection & Evaluation on Held-out Test Split
                 │
                 ▼
    Dual-Task Production Inference & Streamlit Web Deployment
    ```

    ### 4. Benchmark Datasets & Verifiable References
    1. **Task A (Sentiment Analysis)**:
       - *DravidianLangTech / NAACL 2021*: "Overview of the Shared Task on Sentiment Analysis for Dravidian Languages in Code-Mixed Text", EACL/NAACL Anthology.
       - Gold standard corpus mapping positive, negative, and neutral states, augmented with balanced native Tamil domain declarations.
    2. **Task B (News Authenticity)**:
       - *SPELLL 2022 / Springer CCIS*: "A Novel Dataset for Fake News Detection in Tamil Regional Language", Communications in Computer and Information Science, Springer.
       - 5,273 authentic Tamil news headlines annotated for authenticity (Authenticity 0: Real News, Authenticity 1: Fake News).
    3. **IndicBERT Foundation Model**:
       - Kakwani et al. (2020): *"IndicNLPSuite: Monolingual Corpus, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages"*, Findings of EMNLP 2020.
    """)
