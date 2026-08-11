# Movie Sentiment RNN

Movie Sentiment RNN is a Deep Learning CA1 Part B project for movie review sentiment classification and rating prediction. It keeps the original notebook submission intact while adding a production-minded project layout, reusable Python modules, tests, CI, and MLOps documentation.

## Evidence and interpretation

| Evidence-backed measure | Current repository evidence |
| --- | --- |
| Data split | The pipeline produces a leakage-safe **80% train / 10% validation / 10% test** split. |
| Model families | SimpleRNN, LSTM, and GRU architectures support both classification and regression experiments. |

The qualitative outcome is a reproducible recurrent-model comparison with cleaning, augmentation, tokenisation, training, and evaluation. No architecture is labelled “best” without an exact held-out metric and pinned run.

## Academic Context

This project was completed under Singapore Polytechnic, School of Computing, Diploma in Applied AI and Analytics, for the module Deep Learning (ST1504), CA1, Part B.

- Student: Goh Kun Ming, DAAA
- Academic year: AY25/26, Year 2 Semester 1
- Lecturer: Gerald Chua Deng Xiang

## Problem Scope

The project studies recurrent neural network approaches for movie review modelling:

- Binary sentiment classification using SimpleRNN, LSTM, and GRU models.
- Rating prediction as a regression task using recurrent architectures.
- Data cleaning, language-aware preprocessing, augmentation, tokenisation, training, and model comparison.

The final notebook analysis favours classification for the main use case because readers usually need a clear positive or negative review signal more than a continuous score.

## Repository Layout

```text
.
├── notebooks/
│   ├── DELE_CA1_B.ipynb          # Original notebook submission, preserved
│   └── chapters/                 # Generated chapter notebooks split from the original
├── src/movie_sentiment_rnn/      # Reusable project package
├── tests/                        # Pytest coverage for reusable logic
├── scripts/                      # Utility scripts for notebook and repo maintenance
├── data/                         # Local data zones, ignored except .gitkeep
├── models/                       # Local trained model artifacts, ignored except .gitkeep
├── reports/                      # Metrics, figures, and experiment outputs
├── docs/                         # MLOps, data, model, and project documentation
└── .github/workflows/            # CI and security checks
```

See [docs/project_structure.md](docs/project_structure.md) for the full breakdown.

## Quick Start

Create an environment and install the lightweight package:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For notebook execution and model training, install the notebook dependencies:

```bash
python -m pip install -r requirements-notebook.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Place the provided course dataset at:

```text
data/raw/Movie reviews.csv
```

The original notebook still references its original Colab path. For local runs, prefer updating the path to `data/raw/Movie reviews.csv` or using the package utilities in `src/movie_sentiment_rnn`.

## Common Commands

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Run static security checks:

```bash
bandit -r src scripts -ll
```

Generate the split chapter notebooks:

```bash
python scripts/split_notebook.py
```

Inspect a dataset quality report:

```bash
python -m movie_sentiment_rnn quality-report data/raw/"Movie reviews.csv"
```

## MLOps Notes

- Raw data, processed data, trained models, and generated reports are excluded from Git by default.
- CI runs tests, linting, and Bandit security scanning on each push and pull request.
- The package keeps TensorFlow imports lazy so lightweight checks do not need to install or train deep learning models.
- `.env.example` documents all local configuration values without storing secrets.
- `docs/mlops.md` describes the intended experiment lifecycle, artifact naming, tracking, and release gates.

## Notebook Split

The original notebook remains at `notebooks/DELE_CA1_B.ipynb`. Split notebooks are generated into `notebooks/chapters/` by chapter:

1. Imports and setup
2. Data cleaning
3. Exploratory data analysis
4. Data preprocessing
5. Augmented classification models
6. Augmented regression models
7. Non-augmented classification models
8. Non-augmented regression models
9. Model evaluation
10. Method selection and final evaluation

See [docs/notebook_split.md](docs/notebook_split.md) for details.

## Governance

- [CONTRIBUTING.md](CONTRIBUTING.md) explains the development workflow.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) sets collaboration expectations.
- [SECURITY.md](SECURITY.md) explains vulnerability reporting and security checks.

## License

This project is licensed under the [MIT License](LICENSE).
