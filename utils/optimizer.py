import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Optional


@dataclass
class BaseOptimizer(ABC):
    params: np.ndarray
    lr: float

    @abstractmethod
    def update(self, grad, index) -> None:
        pass

    def __call__(self, index=None) -> Union[np.ndarray, float]:
        """現在のパラメータを取得"""
        if index is None:
            return self.params

        return self.params[index]


@dataclass
class SGD(BaseOptimizer):
    def update(
        self,
        grad: Union[float, np.ndarray],
        index: Optional[Union[np.int64, tuple]],
    ) -> None:
        if index is None:
            self.params -= self.lr * grad
        elif isinstance(index, tuple):
            self.params[index[0], index[1]] -= self.lr * grad
        elif isinstance(index, np.int64):
            self.params[index] -= self.lr * grad
        else:
            raise ValueError("index must be int or tuple")


@dataclass
class Adam(BaseOptimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.M = np.zeros(self.params.shape)
        self.V = np.zeros(self.params.shape)

    def update(self, grad: int, index: int) -> None:
        self.M[index] = self.beta1 * self.M[index] + (1 - self.beta1) * grad
        self.V[index] = (
            self.beta2 * self.V[index] + (1 - self.beta2) * grad**2
        )
        M_hat = self.M[index] / (1 - self.beta1)
        V_hat = self.V[index] / (1 - self.beta2)
        self.params[index] -= self.lr * M_hat / ((V_hat**0.5) + self.eps)
