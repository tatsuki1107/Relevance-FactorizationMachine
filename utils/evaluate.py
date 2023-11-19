# Standard library imports
from typing import Tuple, Union, Optional
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party library imports
import numpy as np

# Internal modules imports
from src.mf import LogisticMatrixFactorization as MF
from src.fm import FactorizationMachines as FM
from utils.metrics import metric_candidates
from utils.metrics import calc_ips_of_dcg_at_k

METRIC_NAME_ERROR_MESSAGE = "metric_name must be in {}. metric_name: '{}'"
MODEL_CLASS_ERROR_MESSAGE = "model must be FM, MF or Random. model: '{}'"


@dataclass
class _BaseEvaluator(ABC):
    """モデルごとに評価指標を計算する基底クラス

    Args:
    - _seed (int): 乱数のシード値 (read only)
    - X (Optional, np.ndarray): 特徴量行列。 Noneであれば、ランダム推薦をもって評価する。
    - indices_per_user (list): ユーザーごとに分割されたデータのインデックスのリスト
    - y_true (np.ndarray): ユーザーごとの正解ラベル
    """

    _seed: int
    X: Optional[np.ndarray]
    indices_per_user: list
    y_true: np.ndarray

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    def _predict(self, model: Union[MF, FM, str], indices: list) -> np.ndarray:
        """学習済みモデルを用いてユーザーごとのスコアを予測する

        Args:
        - model (MF,FM,str): 学習済みモデルのインスタンス。または、"Random"文字列。
        - indices (list): 単一ユーザーのデータインデックス.
        Raises:
        - ValueError: modelがMF,FM,Randomのいずれでもない場合

        Returns:
        - y_scores (np.ndarray): ユーザーごとの予測スコア.
        """

        if model.__class__ in {FM, MF}:
            y_scores = model.predict(self.X[indices])

        elif model == "Random":
            np.random.seed(self._seed)
            y_scores = np.random.uniform(0, 1, len(indices))

        else:
            raise ValueError(MODEL_CLASS_ERROR_MESSAGE.format(model))

        return y_scores


@dataclass
class TestEvaluator(_BaseEvaluator):
    """テストデータに対する評価指標を計算するクラス

    Args:
    - K (Tuple[int]): 評価するランキング位置のタプル
    - used_metrics (set): 使用する評価指標の集合
    """

    K: Tuple[int]
    used_metrics: set

    def __post_init__(self) -> None:
        """使用する評価指標の関数を辞書に格納する

        Raises:
        - ValueError: metric_nameが使用可能な評価指標の集合に含まれていない場合
        """

        self.metric_functions = {}
        for metric_name in self.used_metrics:
            if metric_name not in metric_candidates:
                raise ValueError(
                    METRIC_NAME_ERROR_MESSAGE.format(
                        metric_candidates.keys(), metric_name
                    )
                )
            self.metric_functions[metric_name] = metric_candidates[metric_name]

    def evaluate(
        self, model: Union[MF, FM, str], pscores: np.ndarray
    ) -> defaultdict:
        """評価を実行するメソッド

        Args:
        - model (FM,MF,str): 学習済みモデルのインスタンス。または、"Random"文字列。
        - pscores (Optional, np.ndarray): 傾向スコア。

        Returns:
        - results (defaultdict): 評価指標の結果を格納した辞書
        """

        metric_per_user = defaultdict(lambda: defaultdict(list))
        for indices in self.indices_per_user:
            y_scores = self._predict(model, indices)
            ranked_indices = y_scores.argsort()[::-1]
            sorted_y_true = self.y_true[indices][ranked_indices]
            sorted_pscores = pscores[indices][ranked_indices]

            if np.sum(sorted_y_true) == 0:
                continue

            for k in self.K:
                for metric_name, metric_func in self.metric_functions.items():
                    if metric_name == "ME":
                        metric_value = metric_func(sorted_pscores, k)
                    else:
                        metric_value = metric_func(sorted_y_true, k)
                    metric_per_user[metric_name][k].append(metric_value)

        results = defaultdict(list)
        for k in self.K:
            for metric_name in self.metric_functions.keys():
                results[metric_name].append(
                    np.nanmean(metric_per_user[metric_name][k])
                )

        return results


@dataclass
class ValEvaluator(_BaseEvaluator):
    """検証データに対する評価指標を計算するクラス

    Args:
    - k (int): 評価するランキング位置
    - metric_name (str): 使用する評価指標の名前
    """

    k: int
    metric_name: str

    def __post_init__(self) -> None:
        """使用する評価指標の関数を変数に格納する

        Raises:
            ValueError: metric_nameがDCGでない場合。
                        metric.pyの関数を追加する必要がある。
        """

        if self.metric_name == "DCG":
            self.metric_func = calc_ips_of_dcg_at_k
        else:
            raise ValueError("You can use only DCG metric.")

    def evaluate(
        self, model: Union[FM, MF, str], pscores: np.ndarray
    ) -> float:
        """評価を実行するメソッド

        Args:
            model (Union[FM, MF, str]): 学習済みモデルのインスタンス。または、"Random"文字列。
            pscores (np.ndarray): 傾向スコア。

        Returns:
            float: 評価指標の結果
        """

        metric_per_user = []
        for indices in self.indices_per_user:
            y_scores = self._predict(model, indices)
            ranked_indices = y_scores.argsort()[::-1]
            sorted_y_true = self.y_true[indices][ranked_indices]
            sorted_pscores = pscores[indices][ranked_indices]

            if np.sum(sorted_y_true) == 0:
                continue

            metric_value = self.metric_func(
                sorted_y_true, self.k, sorted_pscores
            )
            metric_per_user.append(metric_value)

        return np.mean(metric_per_user)
