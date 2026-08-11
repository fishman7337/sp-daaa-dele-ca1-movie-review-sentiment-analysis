"""Split the source coursework notebook into deterministic section notebooks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "DELE_CA1_B.ipynb"
OUTPUT_DIR = ROOT / "notebooks" / "chapters"

CHAPTERS = [
    ("01_imports_and_setup.ipynb", 1, 6, "Imports and Setup"),
    ("02_data_cleaning.ipynb", 6, 62, "Data Cleaning"),
    ("03_exploratory_data_analysis.ipynb", 62, 102, "Exploratory Data Analysis"),
    ("04_data_preprocessing.ipynb", 102, 177, "Data Preprocessing"),
    ("05_classification_augmented.ipynb", 177, 347, "Classification RNN Models - Augmented"),
    ("06_regression_augmented.ipynb", 347, 517, "Regression RNN Models - Augmented"),
    (
        "07_classification_no_augmentation.ipynb",
        517,
        687,
        "Classification RNN Models - No Augmentation",
    ),
    ("08_regression_no_augmentation.ipynb", 687, 857, "Regression RNN Models - No Augmentation"),
    ("09_model_evaluation.ipynb", 857, 867, "Model Evaluation"),
    (
        "10_method_selection_final_evaluation.ipynb",
        867,
        None,
        "Method Selection and Final Evaluation",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that committed chapter notebooks match the source notebook.",
    )
    return parser.parse_args()


def expected_notebooks() -> dict[Path, dict[str, object]]:
    """Build the expected chapter notebooks without writing to disk.

    Returns:
        Mapping of output paths to JSON-compatible notebook payloads.
    """
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    expected: dict[Path, dict[str, object]] = {}

    for filename, start, stop, title in CHAPTERS:
        chapter = copy.deepcopy(notebook)
        selected = cells[start - 1 : None if stop is None else stop - 1]
        chapter["cells"] = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "Generated from `notebooks/DELE_CA1_B.ipynb` by `scripts/split_notebook.py`.\n",
                ],
            },
            *copy.deepcopy(selected),
        ]
        chapter["metadata"] = copy.deepcopy(notebook.get("metadata", {}))
        expected[OUTPUT_DIR / filename] = chapter
    return expected


def serialise_notebook(notebook: dict[str, object]) -> str:
    """Serialise a notebook using the repository's stable JSON format.

    Args:
        notebook: JSON-compatible notebook payload.

    Returns:
        Deterministically formatted notebook JSON.
    """
    return json.dumps(notebook, ensure_ascii=False, indent=1)


def notebook_matches(path: Path, expected: dict[str, object]) -> bool:
    """Return whether a generated notebook is semantically current.

    Args:
        path: Generated notebook path to inspect.
        expected: Expected JSON-compatible notebook payload.

    Returns:
        ``True`` when the notebook JSON matches regardless of formatting or line endings.
    """
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def main() -> int:
    """Regenerate or verify chapter notebooks and return an exit status."""
    args = parse_args()
    expected = expected_notebooks()

    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, notebook in expected.items()
            if not notebook_matches(path, notebook)
        ]
        if stale:
            print("Stale chapter notebooks: " + ", ".join(stale))
            return 1
        print(f"Split notebooks are up to date ({len(expected)} files).")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, notebook in expected.items():
        path.write_text(serialise_notebook(notebook), encoding="utf-8")

    print(f"Wrote {len(expected)} chapter notebooks to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
