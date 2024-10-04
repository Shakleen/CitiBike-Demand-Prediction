import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period: int):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        sin_transform = np.sin(2 * np.pi * X / self.period)
        cos_transform = np.cos(2 * np.pi * X / self.period)
        return np.concatenate([sin_transform, cos_transform], axis=1)
