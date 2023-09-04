import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class Adam:
    params: np.ndarray
    lr: float
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

    def __call__(self, index=None) -> Union[np.ndarray, float]:
        """現在のパラメータを取得"""
        if index is None:
            return self.params

        return self.params[index]
