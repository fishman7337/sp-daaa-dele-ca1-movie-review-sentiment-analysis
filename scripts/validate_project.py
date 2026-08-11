"""Validate the required structure and integrity of the movie sentiment project."""

from __future__ import annotations

from pathlib import Path

REQUIRED_PATHS = [
    "notebooks/DELE_CA1_B.ipynb",
    "README.md",
    "LICENSE",
    ".env.example",
    "pyproject.toml",
    "src/movie_sentiment_rnn",
    "tests",
    ".github/workflows/ci.yml",
    ".github/README.md",
    ".github/workflows/README.md",
    "data/README.md",
    "data/raw/README.md",
    "data/interim/README.md",
    "data/processed/README.md",
    "docs/README.md",
    "models/README.md",
    "notebooks/README.md",
    "notebooks/chapters/README.md",
    "reports/README.md",
    "reports/figures/README.md",
    "reports/metrics/README.md",
    "scripts/README.md",
    "src/README.md",
    "src/movie_sentiment_rnn/README.md",
    "tests/README.md",
]


def main() -> int:
    """Run repository validation and return a command-line exit status."""
    missing = [path for path in REQUIRED_PATHS if not Path(path).exists()]
    if missing:
        for path in missing:
            print(f"Missing required project path: {path}")
        return 1
    print("Project structure validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
