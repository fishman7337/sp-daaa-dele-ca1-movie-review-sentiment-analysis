"""Regression and classification evaluation metrics for sentiment models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate RMSE, MAE, and coefficient of determination."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(true, pred)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mape": float(mean_absolute_percentage_error(true, pred)),
        "r2": float(r2_score(true, pred)),
    }


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return regression metrics as a presentation-ready dataframe."""
    metrics = regression_metrics(y_true, y_pred)
    return pd.DataFrame(
        {
            "Metric": ["MAE", "MSE", "RMSE", "MAPE", "R2"],
            "Value": [
                metrics["mae"],
                metrics["mse"],
                metrics["rmse"],
                metrics["mape"],
                metrics["r2"],
            ],
        }
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 for binary labels."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
    }
