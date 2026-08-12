"""Class-balancing and score-band augmentation helpers."""

from __future__ import annotations

import pandas as pd


def upsample_to_largest_class(
    frame: pd.DataFrame,
    label_column: str,
    *,
    random_state: int = 42,
    source_column: str = "source",
) -> pd.DataFrame:
    """Balance labels by sampling minority classes with replacement."""
    counts = frame[label_column].value_counts()
    if counts.empty:
        return frame.copy()

    target_size = int(counts.max())
    groups = []
    for _, group in frame.groupby(label_column, sort=False):
        base = group.copy()
        if source_column not in base.columns:
            base[source_column] = "original"
        base[source_column] = base[source_column].fillna("original")

        needed = target_size - len(base)
        if needed <= 0:
            groups.append(base.sample(n=target_size, random_state=random_state))
            continue

        extra = base.sample(n=needed, replace=True, random_state=random_state).copy()
        extra[source_column] = "augmented"
        groups.append(pd.concat([base, extra], ignore_index=True))

    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=random_state)


def score_band(score: float) -> str:
    """Map a normalized score to its low, mid, or high value band."""
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "mid"
    return "high"


def add_score_band(frame: pd.DataFrame, score_column: str = "Score") -> pd.DataFrame:
    """Return a dataframe copy with a derived ``Score_Band`` column."""
    banded = frame.copy()
    banded["Score_Band"] = banded[score_column].apply(score_band)
    return banded
