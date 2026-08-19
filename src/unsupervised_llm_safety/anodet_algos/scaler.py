"""Scaler classes for anomaly detection algorithms.

This module provides scaler implementations for preprocessing data
in various anomaly detection methods.
"""


class IdentityScaler:
    """A scaler that does nothing - for methods that don't need standardization.

    This scaler is useful for:
    - CVDD L0 with binary data
    - CVDD KMeans Masked with raw input data
    - Any method that works directly on raw/unscaled data

    It provides a sklearn-compatible interface for consistency.
    """

    def fit(self, X):
        """Fit the scaler (no-op for identity scaler).

        Args:
            X: Input data (ignored)

        Returns:
            self
        """
        return self

    def transform(self, X):
        """Transform the data (returns as-is for identity scaler).

        Args:
            X: Input data

        Returns:
            X unchanged
        """
        return X

    def fit_transform(self, X):
        """Fit and transform the data (returns as-is for identity scaler).

        Args:
            X: Input data

        Returns:
            X unchanged
        """
        return X

    # For compatibility with sklearn scalers
    mean_ = None
    scale_ = None
