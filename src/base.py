from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class PointwiseBaseRecommender(ABC):
    """ポイントワイズ損失を用いた推薦アルゴリズムの基底クラス

    Args:
        n_epochs: 学習エポック数
        n_factors: 因子行列の次元数
        lr: 学習率
        batch_size: バッチサイズ
        seed: パラメータ初期化のシード値
    """

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
        """与えられたデータを元にクロスエントロピー損失を計算する

        Args:
        - y_trues (np.ndarray): 正解ラベルの配列
        - y_scores (np.ndarray): 予測確率の配列
        - pscores (np.ndarray): 傾向スコアの配列
        - eps (float): ゼロ除算を防ぐための微小値

        Returns:
        - logloss (float): クロスエントロピー損失
        """
        logloss = -np.sum(
            (y_trues / pscores) * np.log(y_scores + eps)
            + (1 - y_trues / pscores) * np.log(1 - y_scores + eps)
        ) / len(y_trues)

        return logloss

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """シグモイド関数。予測値に対してシグモイド関数を適用することで確率に変換する。オーバーフローを防ぐためにクリッピングを行う。

        Args:
            x: 予測値

        Returns:
            np.ndarray: 確率
        """
        x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))
