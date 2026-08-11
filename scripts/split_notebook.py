"""Split the source coursework notebook into deterministic section notebooks."""

from __future__ import annotations

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


def main() -> int:
    """Regenerate section notebooks and return a command-line exit status."""
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        (OUTPUT_DIR / filename).write_text(
            json.dumps(chapter, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    print(f"Wrote {len(CHAPTERS)} chapter notebooks to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
