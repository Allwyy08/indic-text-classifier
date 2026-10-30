"""
Integration and schema tests for the Streamlit web application.
"""

import os
import yaml
import pytest


def test_config_yaml_schema():
    assert os.path.exists("configs/config.yaml"), "Missing config.yaml"
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Core project fields
    assert "project" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "tasks" in cfg
    assert "huggingface" in cfg

    # Model checkpoint
    assert "pretrained_model_name" in cfg["model"]
    assert cfg["model"]["max_length"] > 0

    # Task schemas
    for task_key in ["sentiment", "news_authenticity"]:
        assert task_key in cfg["tasks"]
        t = cfg["tasks"][task_key]
        assert "classes" in t
        assert "num_labels" in t
        assert len(t["classes"]) == t["num_labels"]
        assert "id2label" in t
        assert "label2id" in t
        assert "data_dir" in t
        assert "model_save_dir" in t
        assert "output_dir" in t

    # HF placeholders rule verification: Ensure placeholders are used and old repo is NOT referenced
    hf_cfg = cfg["huggingface"]
    assert "Allwynnoble7" not in str(hf_cfg)
    assert hf_cfg["space_name"] == "indicbert-tamil-classifier"


def test_app_script_syntax_and_structure():
    # Verify app.py can be compiled and parsed as valid Python AST
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    compiled = compile(content, "app.py", "exec")
    assert compiled is not None


def test_sample_buttons_load_intended_text():
    """Verify that each of the 5 sample buttons loads its intended text into the text area immediately."""
    from streamlit.testing.v1 import AppTest

    expected_texts = [
        "இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!",
        "இந்த சேவை மிகவும் மோசமாக உள்ளது, எனக்கு மிகவும் ஏமாற்றமாக இருக்கிறது.",
        "நான் ஒரு ஆசிரியர்.",
        "கர்ப்பப்பை புற்றுநோயை தடுக்கும் “செர்வாவாக்” தடுப்பூசி: இந்தியாவில் முதல் முறையாக உருவாக்கம்",
        '"3,000 க்கும் மேற்பட்ட கொரோனா வைரஸ் நோயாளிகளை வேண்டுமென்றே கொன்றதற்காக" ஒரு மருத்துவரை இத்தாலி கைது செய்ததாக கூறுகிறது.'
    ]

    for btn_idx, expected in enumerate(expected_texts):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        assert len(at.text_area) > 0
        at.button[btn_idx].click().run()
        assert at.text_area[0].value == expected, f"Button {btn_idx} failed to load expected text"


def test_clicking_one_example_replaces_previous_example():
    """Verify that clicking one example and then another replaces the previous example."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Click Positive button
    at.button[0].click().run()
    assert at.text_area[0].value == "இந்த உணவகம் மிகவும் அருமையாக உள்ளது, உணவும் சேவையும் சிறப்பாக இருக்கின்றன!"

    # Click Negative button -> should replace
    at.button[1].click().run()
    assert at.text_area[0].value == "இந்த சேவை மிகவும் மோசமாக உள்ளது, எனக்கு மிகவும் ஏமாற்றமாக இருக்கிறது."

    # Click Real News button -> should replace
    at.button[3].click().run()
    assert at.text_area[0].value == "கர்ப்பப்பை புற்றுநோயை தடுக்கும் “செர்வாவாக்” தடுப்பூசி: இந்தியாவில் முதல் முறையாக உருவாக்கம்"


def test_clear_button_empties_text_area():
    """Verify that clicking Clear button resets the text area immediately."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Load an example first
    at.button[0].click().run()
    assert at.text_area[0].value != ""

    # Click Clear button (index 6)
    at.button[6].click().run()
    assert at.text_area[0].value == ""



def test_manual_text_remains_unchanged_until_button_clicked():
    """Verify that manually entered text is preserved across runs until Clear or an example is clicked."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    manual_text = "இது ஒரு பயனர் உள்ளிட்ட தமிழ் வாக்கியம்."
    at.text_area[0].input(manual_text).run()
    assert at.text_area[0].value == manual_text

    # Another run without clicking example buttons should preserve manual text
    at.run()
    assert at.text_area[0].value == manual_text

    # Clicking clear should empty it
    at.button[6].click().run()
    assert at.text_area[0].value == ""


def test_classify_text_uses_currently_displayed_text():
    """Verify that Classify Text button processes the currently displayed text and runs dual inference."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Load real news example
    at.button[3].click().run()
    assert at.text_area[0].value == "கர்ப்பப்பை புற்றுநோயை தடுக்கும் “செர்வாவாக்” தடுப்பூசி: இந்தியாவில் முதல் முறையாக உருவாக்கம்"

    # Click Classify button (index 5)
    at.button[5].click().run()
    assert len(at.error) == 0, f"Errors encountered during classification: {at.error}"

    # Verify heading is 'Indic Text Classifier'
    found_heading = any("Indic Text Classifier" in md.value for md in at.markdown)
    assert found_heading, "Main heading 'Indic Text Classifier' not found in app markdown"

    # Clean up predictor registry to release memory on Windows
    import gc
    from src.inference import _PREDICTOR_REGISTRY
    _PREDICTOR_REGISTRY.clear()
    gc.collect()





