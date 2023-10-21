# Standard library imports
from typing import Tuple, Union, Optional
from collections import defaultdict
from dataclasses import dataclass

# Third-party library imports
import numpy as np

# Internal modules imports
from src.mf import LogisticMatrixFactorization as MF
from src.fm import FactorizationMachines as FM
from utils.metrics import metric_candidates

METRIC_NAME_ERROR_MESSAGE = "metric_name must be in {}. metric_name: '{}'"
MODEL_CLASS_ERROR_MESSAGE = "model must be FM, MF or Random. model: '{}'"


@dataclass
class Evaluator:
    """モデルごとに評価指標を計算するクラス

    Args:
    - X (Optional, np.ndarray): 特徴量行列。 Noneであれば、ランダム推薦をもって評価する。
    - indices_per_user (list): ユーザーごとに分割されたデータのインデックスのリスト
    - used_matrics (set): 使用する評価指標の集合
    - K (Tuple[int]): 評価するランキング位置のタプル
    - thetahold (Optional, float): 二値分類の閾値
    """

    X: Optional[np.ndarray]
    y_true: np.ndarray
    indices_per_user: list
    used_metrics: set
    K: Tuple[int] = (1, 3, 5)
    thetahold: Optional[float] = 0.75

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
        self, model: Union[MF, FM, str], pscores: Optional[np.ndarray] = None
    ) -> defaultdict:
        """評価を実行するメソッド

        Args:
        - model (FM,MF,str): 学習済みモデルのインスタンス。または、"Random"文字列。
        - pscores (Optional, np.ndarray): 傾向スコア。検証時に引数として必要。

        Returns:
        - results (defaultdict): 評価指標の結果を格納した辞書
        """

        if pscores is None:
            pscores = np.ones_like(self.y_true)

        metric_per_user = defaultdict(lambda: defaultdict(list))
        for indices in self.indices_per_user:
            y_scores = self._predict(model, indices)
            y_true = self.y_true[indices]
            user_pscores = pscores[indices]

            for k in self.K:
                for metric_name, metric_func in self.metric_functions.items():
                    metric_per_user[metric_name][k].append(
                        metric_func(y_true, y_scores, k, user_pscores)
                    )

        results = defaultdict(list)
        for k in self.K:
            for metric_name in self.metric_functions.keys():
                results[metric_name].append(
                    np.nanmean(metric_per_user[metric_name][k])
                )

        return results

    def _predict(
        self,
        model: Union[MF, FM, str],
        indices: list = None,
    ) -> np.ndarray:
        """学習済みモデルを用いてユーザーごとのスコアを予測する

        Args:
        - model (MF,FM,str): 学習済みモデルのインスタンス。または、"Random"文字列。
        - indices (list): 単一ユーザーのデータインデックス. Noneであれば、
                    全ユーザーのデータを用いて予測する。
        Raises:
        - ValueError: modelがMF,FM,Randomのいずれでもない場合

        Returns:
        - y_scores (np.ndarray): ユーザーごとのスコア.閾値があればバイナリ化する。
        """

        if indices is None:
            indices = np.arange(len(self.y_true))

        if model.__class__ in {FM, MF}:
            y_scores = model.predict(self.X[indices])

        elif model == "Random":
            y_scores = np.random.randint(0, 2, len(indices))

        else:
            raise ValueError(MODEL_CLASS_ERROR_MESSAGE.format(model))

        if self.thetahold is None:
            return y_scores

        return (y_scores >= self.thetahold).astype(int)
