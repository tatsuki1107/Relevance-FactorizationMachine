# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Third-party library imports
import numpy as np


@dataclass
class PointwiseBaseRecommender(ABC):
    """base class for pointwise recommenders

    Args:
    - estimator (str): estimator name
    - n_epochs (int): number of epochs
    - n_factors (int): number of factors
    - lr (float): learning rate
    - batch_size (int): batch size
    - seed (int): random seed
    """

    estimator: str
    n_epochs: int
    n_factors: int
    lr: float
    batch_size: int
    seed: int

    @abstractmethod
    def fit(self, train, val) -> tuple:
        pass

    @abstractmethod
    def predict(self, **kwargs) -> np.ndarray:
        pass

    def _cross_entropy_loss(
        self,
        y_trues: np.ndarray,
        y_scores: np.ndarray,
        pscores: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        """calculate cross entropy loss

        Args:
        - y_trues (np.ndarray): true labels
        - y_scores (np.ndarray): predicted scores
        - pscores (np.ndarray): propensity scores
        - eps (float, optional): small value to avoid overflow. Defaults to 1e-8.

        Returns:
            float: cross entropy loss
        """

        logloss = -np.sum(
            (y_trues / pscores) * np.log(y_scores + eps)
            + (1 - y_trues / pscores) * np.log(1 - y_scores + eps)
        ) / len(y_trues)

        return logloss

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """sigmoid function"""
        x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))
