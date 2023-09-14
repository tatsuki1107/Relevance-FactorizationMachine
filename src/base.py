from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class PointwiseBaseRecommender(ABC):
    n_epochs: int
    n_factors: int
    scale: float
    lr: float
    batch_size: int
    seed: int

    @abstractmethod
    def fit(self, train, val, test) -> tuple:
        pass

    @abstractmethod
    def predict(self, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def _cross_entropy_loss(self, **kwargs) -> float:
        pass

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
