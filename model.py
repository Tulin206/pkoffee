# filepath: /Users/tulin20/Documents/s3_school/pkoffee/model.py

"""Lightweight OOP Model utilities for the project.

This module provides a simple Model class that can:
- load CSV data into a pandas DataFrame
- train a linear regression using numpy (least squares)
- make predictions on new data
- evaluate RMSE
- save and load the trained model using pickle

The implementation keeps dependencies minimal (pandas, numpy, pickle).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass
class Model:
    """Simple model wrapper using ordinary least squares linear regression.

    Attributes:
        data: loaded pandas DataFrame (optional)
        feature_cols: list of feature column names used for training
        target_col: target column name used for training
        coefficients: numpy array of learned coefficients (excluding intercept)
        intercept: learned intercept (float)
    """

    data: Optional[pd.DataFrame] = None
    feature_cols: Optional[List[str]] = None
    target_col: Optional[str] = None
    coefficients: Optional[np.ndarray] = None
    intercept: Optional[float] = None

    def load_data(self, path: str, **read_csv_kwargs) -> pd.DataFrame:
        """Load CSV data from `path` into the `data` attribute and return it."""
        self.data = pd.read_csv(path, **read_csv_kwargs)
        return self.data

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first `n` rows of the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.data.head(n)

    def describe(self) -> pd.DataFrame:
        """Return a description (pandas describe) of the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.data.describe()

    def _prepare_X_y(self, target_col: str, feature_cols: Optional[Sequence[str]] = None):
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if target_col not in self.data.columns:
            raise KeyError(f"Target column '{target_col}' not found in data")

        y = self.data[target_col].to_numpy(dtype=float)

        if feature_cols is None:
            # default to all numeric columns except the target
            feature_cols = [c for c in self.data.select_dtypes(include=[np.number]).columns if c != target_col]

        X = self.data[list(feature_cols)].to_numpy(dtype=float)
        return X, y, list(feature_cols)

    def train_linear(self, target_col: str, feature_cols: Optional[Sequence[str]] = None) -> None:
        """Train a linear regression by ordinary least squares.

        After calling this method, `coefficients`, `intercept`, `feature_cols`, and `target_col`
        are populated on the instance.
        """
        X, y, feature_cols = self._prepare_X_y(target_col, feature_cols)

        # Add intercept column (ones)
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_with_intercept = np.hstack([ones, X])

        # Solve least squares: theta = (X^T X)^{-1} X^T y
        theta, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

        self.intercept = float(theta[0])
        self.coefficients = np.asarray(theta[1:], dtype=float)
        self.feature_cols = feature_cols
        self.target_col = target_col

    def predict(self, X: Union[pd.DataFrame, Sequence[Sequence[float]], Sequence[float]]) -> np.ndarray:
        """Predict target values for given features.

        X may be a pandas DataFrame (with columns matching `feature_cols`) or a 2D/1D sequence.
        """
        if self.coefficients is None or self.intercept is None or self.feature_cols is None:
            raise ValueError("Model is not trained. Call train_linear() first.")

        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise KeyError(f"Missing feature columns in input DataFrame: {missing}")
            X_mat = X[self.feature_cols].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                # single sample with same length as number of features
                if arr.shape[0] != len(self.feature_cols):
                    raise ValueError("Input length does not match number of feature columns")
                X_mat = arr.reshape(1, -1)
            else:
                X_mat = arr

        return (X_mat @ self.coefficients) + self.intercept

    @staticmethod
    def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    def evaluate(self) -> float:
        """Evaluate the trained model on the training data and return RMSE."""
        if self.data is None:
            raise ValueError("No data loaded.")
        if self.target_col is None:
            raise ValueError("No trained model. Call train_linear() first.")

        X = self.data[self.feature_cols]
        y_true = self.data[self.target_col]
        y_pred = self.predict(X)
        return self.rmse(y_true, y_pred)

    def save(self, path: str) -> None:
        """Serialize the whole Model instance to `path` using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a previously saved Model instance from `path`."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Model):
            raise TypeError("Pickle file does not contain a Model instance")
        return obj


# Example usage (not executed on import):
# m = Model()
# m.load_data('coffee_productivity.csv')
# m.train_linear(target_col='productivity', feature_cols=['hours_slept', 'cups_of_coffee'])
# print('RMSE on training data:', m.evaluate())
# m.save('trained_model.pkl')
