"""Dataset loading, validation, cleaning, and deterministic split helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REVIEW_COLUMN = "Review"
SCORE_COLUMN = "Score"
SOURCE_COLUMN = "source"
DEFAULT_MISC_COLUMNS = (
    "Are there ways for you to generate more data? Spliting up sentences, would that help?",
)


@dataclass(frozen=True)
class DataQualityReport:
    """Summary counts describing review-dataset quality."""

    row_count: int
    column_count: int
    missing_by_column: dict[str, int]
    duplicate_rows: int
    short_reviews: int

    def as_dict(self) -> dict[str, object]:
        """Return the report as a JSON-compatible mapping."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_by_column": self.missing_by_column,
            "duplicate_rows": self.duplicate_rows,
            "short_reviews": self.short_reviews,
        }


def load_reviews_csv(path: str | Path) -> pd.DataFrame:
    """Load a review CSV and verify its required columns."""
    frame = pd.read_csv(path)
    validate_review_frame(frame)
    return frame


def validate_review_frame(
    frame: pd.DataFrame,
    required_columns: Iterable[str] = (REVIEW_COLUMN, SCORE_COLUMN),
) -> None:
    """Raise ``ValueError`` when required review columns are absent."""
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Dataset is missing required column(s): {joined}")


def build_quality_report(frame: pd.DataFrame, min_words: int = 4) -> DataQualityReport:
    """Count rows, missing values, duplicates, and unusually short reviews."""
    validate_review_frame(frame)
    review_lengths = frame[REVIEW_COLUMN].fillna("").astype(str).str.split().str.len()
    return DataQualityReport(
        row_count=len(frame),
        column_count=len(frame.columns),
        missing_by_column={str(col): int(count) for col, count in frame.isna().sum().items()},
        duplicate_rows=int(frame.duplicated().sum()),
        short_reviews=int((review_lengths < min_words).sum()),
    )


def drop_misc_columns(
    frame: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_MISC_COLUMNS,
) -> pd.DataFrame:
    """Return a copy without known export-only metadata columns."""
    return frame.drop(columns=[col for col in columns if col in frame.columns])


def remove_missing_scores(frame: pd.DataFrame, score_column: str = SCORE_COLUMN) -> pd.DataFrame:
    """Remove rows whose sentiment score is missing."""
    return frame[frame[score_column].notna()].reset_index(drop=True)


def remove_duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and reset the index."""
    return frame.drop_duplicates().reset_index(drop=True)


def remove_short_reviews(
    frame: pd.DataFrame,
    review_column: str = REVIEW_COLUMN,
    min_words: int = 4,
) -> pd.DataFrame:
    """Remove reviews shorter than the configured word threshold."""
    mask = frame[review_column].fillna("").astype(str).str.split().str.len() >= min_words
    return frame[mask].reset_index(drop=True)


def invert_scores(frame: pd.DataFrame, score_column: str = SCORE_COLUMN) -> pd.DataFrame:
    """Invert normalized scores so larger values represent positive sentiment."""
    cleaned = frame.copy()
    cleaned[score_column] = 1 - pd.to_numeric(cleaned[score_column], errors="raise")
    return cleaned


def assign_binary_sentiment(
    frame: pd.DataFrame,
    score_column: str = SCORE_COLUMN,
    label_column: str = "Sentiment_Label",
    target_column: str = "Sentiment_Binary",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Derive a binary sentiment label from a normalized score threshold."""
    labelled = frame.copy()
    positive_mask = pd.to_numeric(labelled[score_column], errors="raise") >= threshold
    labelled[label_column] = np.where(positive_mask, "Positive", "Negative")
    labelled[target_column] = np.where(positive_mask, 1, 0)
    return labelled


def add_source_column(frame: pd.DataFrame, source: str = "original") -> pd.DataFrame:
    """Tag every review with its provenance label."""
    sourced = frame.copy()
    sourced[SOURCE_COLUMN] = source
    return sourced


def clean_review_frame(
    frame: pd.DataFrame,
    *,
    min_words: int = 4,
    invert_score: bool = True,
    add_sentiment: bool = True,
) -> pd.DataFrame:
    """Run the deterministic review-cleaning pipeline on a dataframe copy."""
    validate_review_frame(frame)
    cleaned = drop_misc_columns(frame)
    cleaned = remove_missing_scores(cleaned)
    cleaned = remove_duplicate_rows(cleaned)
    cleaned = remove_short_reviews(cleaned, min_words=min_words)
    if invert_score:
        cleaned = invert_scores(cleaned)
    if add_sentiment:
        cleaned = assign_binary_sentiment(cleaned)
    return add_source_column(cleaned)


def train_validation_test_split(
    frame: pd.DataFrame,
    *,
    target_column: str | None = None,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train, validation, and test partitions."""
    stratify = frame[target_column] if target_column else None
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    relative_validation_size = validation_size / (1 - test_size)
    stratify_train_val = train_val[target_column] if target_column else None
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation_size,
        stratify=stratify_train_val,
        random_state=random_state,
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
