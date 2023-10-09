import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Optional


@dataclass
class BaseOptimizer(ABC):
    """最適化アルゴリズムの基底クラス

    Args:
    - params (np.ndarray): 最適化するパラメータ
    - lr (float): 学習率
    """

    params: np.ndarray
    lr: float

    @abstractmethod
    def update(
        self,
        grad: Union[float, np.ndarray],
        index: Optional[Union[int, tuple]],
    ) -> None:
        """与えられた勾配を使用してパラメータを更新する。

        Args:
        - grad (Union[float, np.ndarray]): パラメータの勾配。
        - index (Optional[Union[int, tuple]]): パラメータ更新のインデックス。
        """
        pass

    def __call__(self, index=None) -> Union[np.ndarray, float]:
        """現在のパラメータを取得する。

        Args:
        - index (Optional[int, tuple]): 取得するパラメータのインデックス。

        Returns:
        - (Union[np.ndarray, float]): 現在のパラメータまたは特定のパラメータ値。
        """

        if index is None:
            return self.params

        return self.params[index]


@dataclass
class SGD(BaseOptimizer):
    """確率的勾配降下法 (Stochastic Gradient Descent) による最適化クラス"""

    def update(
        self,
        grad: Union[float, np.ndarray],
        index: Optional[Union[int, tuple]],
    ) -> None:
        """SGDを使用してパラメータを更新する。

        Args:
        - grad (Union[float, np.ndarray]): パラメータの勾配。
        - index (Optional[Union[int, tuple]]): パラメータ更新のインデックス。
        """
        if index is None:
            self.params -= self.lr * grad
        else:
            self.params[index] -= self.lr * grad


@dataclass
class Adam(BaseOptimizer):
    """Adamによる最適化クラス。

    Args:
    - beta1 (float): 第一モーメントの指数的減衰率。デフォルトは0.9。
    - beta2 (float): 第二モーメントの指数的減衰率。デフォルトは0.999。
    - eps (float): ゼロ除算を防ぐための小さな値。デフォルトは1e-8。
    """

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """第一および第二モーメントのベクトルを初期化しする。"""
        self.M = np.zeros(self.params.shape)
        self.V = np.zeros(self.params.shape)

    def update(self, grad: int, index: int) -> None:
        """Adamを使用してパラメータを更新する。

        Args:
        - grad (int): パラメータの勾配。
        - index (int): パラメータ更新のインデックス。
        """

        self.M[index] = self.beta1 * self.M[index] + (1 - self.beta1) * grad
        self.V[index] = (
            self.beta2 * self.V[index] + (1 - self.beta2) * grad**2
        )
        M_hat = self.M[index] / (1 - self.beta1)
        V_hat = self.V[index] / (1 - self.beta2)
        self.params[index] -= self.lr * M_hat / ((V_hat**0.5) + self.eps)
