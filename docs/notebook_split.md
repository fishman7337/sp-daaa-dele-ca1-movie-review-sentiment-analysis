# Notebook Split

`notebooks/DELE_CA1_B.ipynb` remains the original notebook submission. Chapter notebooks are generated for easier review and maintenance.

Generate them with:

```bash
python scripts/split_notebook.py
```

Verify them without modifying files:

```bash
python scripts/split_notebook.py --check
```

## Generated Chapters

| File | Original section |
| --- | --- |
| `01_imports_and_setup.ipynb` | Title and imports |
| `02_data_cleaning.ipynb` | Data cleaning |
| `03_exploratory_data_analysis.ipynb` | Exploratory data analysis |
| `04_data_preprocessing.ipynb` | Data preprocessing |
| `05_classification_augmented.ipynb` | Classification with augmentation |
| `06_regression_augmented.ipynb` | Regression with augmentation |
| `07_classification_no_augmentation.ipynb` | Classification without augmentation |
| `08_regression_no_augmentation.ipynb` | Regression without augmentation |
| `09_model_evaluation.ipynb` | Model comparison |
| `10_method_selection_final_evaluation.ipynb` | Final method selection and bibliography |

## Regeneration Policy

The split notebooks are derived artifacts. If `notebooks/DELE_CA1_B.ipynb` changes, regenerate the chapter notebooks and review the diff. CI runs the check command so stale chapter notebooks cannot pass unnoticed.
