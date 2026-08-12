"""Environment-backed configuration for data, model, and report artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_available(env_file: str | Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(env_file)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration loaded from environment variables."""

    data_raw_path: Path = Path("data/raw/Movie reviews.csv")
    data_processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    random_seed: int = 42
    max_words: int = 10000
    max_sequence_length: int = 100
    test_size: float = 0.10
    validation_size: float = 0.10
    mlflow_tracking_uri: str | None = None
    experiment_name: str = "movie-sentiment-rnn"

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> ProjectConfig:
        """Build a project configuration from an optional environment file."""
        _load_dotenv_if_available(env_file)
        return cls(
            data_raw_path=Path(os.getenv("DATA_RAW_PATH", "data/raw/Movie reviews.csv")),
            data_processed_dir=Path(os.getenv("DATA_PROCESSED_DIR", "data/processed")),
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            report_dir=Path(os.getenv("REPORT_DIR", "reports")),
            random_seed=_int_env("RANDOM_SEED", 42),
            max_words=_int_env("MAX_WORDS", 10000),
            max_sequence_length=_int_env("MAX_SEQUENCE_LENGTH", 100),
            test_size=_float_env("TEST_SIZE", 0.10),
            validation_size=_float_env("VALIDATION_SIZE", 0.10),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI") or None,
            experiment_name=os.getenv("EXPERIMENT_NAME", "movie-sentiment-rnn"),
        )

    def ensure_artifact_dirs(self) -> None:
        """Create the configured processed-data, model, and report directories."""
        for path in (self.data_processed_dir, self.model_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)
