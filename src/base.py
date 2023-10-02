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
    scale: float
    lr: float
    batch_size: int
    seed: int

    @abstractmethod
    def fit(self, train, val) -> tuple:
        """モデルの学習を行う
        Args:
            train: 学習データ
            val: 検証データ

        Returns:
            tuple: 学習データと検証データの損失履歴
        """
        pass

    @abstractmethod
    def predict(self, **kwargs) -> np.ndarray:
        """引数として与えられた特徴量に対する予測値を返す

        Returns:
            np.ndarray: 予測値の配列
        """
        pass

    @abstractmethod
    def _cross_entropy_loss(self, **kwargs) -> float:
        """与えられたデータを元にクロスエントロピー損失を計算する

        Returns:
            float: クロスエントロピー損失
        """
        pass

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """シグモイド関数。予測値に対してシグモイド関数を適用することで確率に変換する。オーバーフローを防ぐためにクリッピングを行う。

        Args:
            x: 予測値

        Returns:
            np.ndarray: 確率
        """
        x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))
