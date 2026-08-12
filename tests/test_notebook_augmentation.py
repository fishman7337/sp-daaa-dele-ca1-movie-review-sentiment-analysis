import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from movie_sentiment_rnn.augmentation import score_band

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "DELE_CA1_B.ipynb"
SETUP_CHAPTER = ROOT / "notebooks" / "chapters" / "01_imports_and_setup.ipynb"
PREPROCESSING_CHAPTER = ROOT / "notebooks" / "chapters" / "04_data_preprocessing.ipynb"


def _load_split_notebook() -> ModuleType:
    path = ROOT / "scripts" / "split_notebook.py"
    spec = importlib.util.spec_from_file_location("movie_split_notebook", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load notebook splitter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


notebook_matches = _load_split_notebook().notebook_matches


def _cells(path: Path, cell_type: str | None = None) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell_type is None or cell["cell_type"] == cell_type
    ]


def test_regression_augmentation_includes_true_high_score_band() -> None:
    source = "\n".join(_cells(SOURCE_NOTEBOOK, "code"))

    assert "df_low = df_reg_train[df_reg_train['Score'] < 0.4]" in source
    assert "df_high = df_reg_train[df_reg_train['Score'] >= 0.7]" in source
    assert "for df_bin in [df_low, df_mid, df_high]:" in source
    assert "return 'Low'\n    elif score < 0.7:" in source
    assert "else:\n        return 'High'" in source


def test_reusable_score_band_helper_matches_notebook_boundaries() -> None:
    assert score_band(0.0) == "low"
    assert score_band(0.3999) == "low"
    assert score_band(0.4) == "mid"
    assert score_band(0.6999) == "mid"
    assert score_band(0.7) == "high"
    assert score_band(1.0) == "high"


def test_augmentation_fallbacks_only_handle_model_io_errors() -> None:
    fallback_cells = [
        source for source in _cells(SOURCE_NOTEBOOK, "code") if "model unavailable" in source
    ]

    assert len(fallback_cells) == 6
    for source in fallback_cells:
        handlers = [
            node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler)
        ]
        assert len(handlers) == 1
        assert isinstance(handlers[0].type, ast.Name)
        assert handlers[0].type.id == "OSError"
        assert "warnings.warn(" in source


def test_augmentation_prose_matches_the_nllb_implementation() -> None:
    markdown = "\n".join(_cells(SOURCE_NOTEBOOK, "markdown"))

    assert "Deep" + "-Translator" not in markdown
    assert markdown.count("up to three generated variants") == 2
    assert markdown.count("up to four times") == 2
    assert "including genuinely high-score samples" in markdown


def test_stochastic_augmentation_uses_the_documented_seed() -> None:
    code = "\n".join(_cells(SOURCE_NOTEBOOK, "code"))
    augmentation_cells = [
        source for source in _cells(SOURCE_NOTEBOOK, "code") if "do_sample=True" in source
    ]

    assert "AUGMENTATION_SEED = 42" in code
    assert "set_seed(AUGMENTATION_SEED)" in code
    assert len(augmentation_cells) == 4
    assert sum(source.count(".sample(") for source in augmentation_cells) == 6
    assert sum(source.count("random_state=AUGMENTATION_SEED") for source in augmentation_cells) == 6


def test_generated_chapters_preserve_augmentation_safeguards() -> None:
    setup_code = "\n".join(_cells(SETUP_CHAPTER, "code"))
    preprocessing = "\n".join(_cells(PREPROCESSING_CHAPTER))

    assert "set_seed(AUGMENTATION_SEED)" in setup_code
    assert "for df_bin in [df_low, df_mid, df_high]:" in preprocessing
    assert preprocessing.count("except OSError as exc:") == 6
    assert "Deep" + "-Translator" not in preprocessing


def test_chapter_drift_check_ignores_json_formatting(tmp_path: Path) -> None:
    expected = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "chapter.ipynb"
    path.write_text(json.dumps(expected, separators=(",", ":")), encoding="utf-8")

    assert notebook_matches(path, expected)


def test_chapter_drift_check_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "chapter.ipynb"
    path.write_text("not-json", encoding="utf-8")

    assert not notebook_matches(path, {})
