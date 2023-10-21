# Standard library imports
from dataclasses import dataclass
from logging import Logger
from typing import Tuple

# Third-party library imports
import numpy as np
import pandas as pd

# Internal modules imports
from conf.config import LogDataPropensityConfig
from utils.dataloader.base import BaseLoader
from utils.dataloader._kuairec import KuaiRecCSVLoader


BEHAVIOR_POLICY_NAME_ERROR_MESSAGE = "behavior_policy must be random."


@dataclass
class SemiSyntheticLogDataGenerator(BaseLoader):
    """半人工的にログデータを生成する

    Args:
    - _seed (int): 乱数シード (read only)
    - _params (LogDataPropensityConfig): 半人工データ生成のパラメータ (read only)
    - logger (Logger): Loggerクラスのインスタンス
    """

    _seed: int
    _params: LogDataPropensityConfig
    logger: Logger

    def load(
        self,
        interaction_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """半人工データの生成を実行するメソッド

        Args:
        - interaction_df (pd.DataFrame): kuairec/small_matrix.csvの
        インタラクションデータ. 評価値行列の密度は100%

        Returns:
        - interaction_df (pd.DataFrame): ランダムポリシーによって生成された半人工データ
        """

        interaction_df = self._exract_data_by_policy(
            interaction_df=interaction_df
        )
        datatypes = self._generate_datatype(
            data_size=interaction_df.shape[0],
        )
        interaction_df["datatype"] = datatypes

        relevance_probabilities = self._generate_relevance(
            watch_ratio=interaction_df["watch_ratio"]
        )
        interaction_df["relevance"] = relevance_probabilities
        interaction_df.drop("watch_ratio", axis=1, inplace=True)
        self.logger.info(
            f"mean of relevance: {relevance_probabilities.mean()}"
        )
        self.logger.info(f"std of relevance: {relevance_probabilities.std()}")

        exposure_probabilities = self._generate_exposure(
            existing_video_ids=interaction_df["video_id"]
        )
        interaction_df["exposure"] = exposure_probabilities
        self.logger.info(f"mean of exposure: {exposure_probabilities.mean()}")
        self.logger.info(f"std of exposure: {exposure_probabilities.std()}")

        biased_clicks, unbiased_clicks = self._generate_clicks(
            exposure_probabilities=interaction_df["exposure"],
            relevance_probabilities=interaction_df["relevance"],
        )
        interaction_df["biased_click"] = biased_clicks
        interaction_df["unbiased_click"] = unbiased_clicks
        self.logger.info(f"biased click through rate: {biased_clicks.mean()}")
        self.logger.info(
            f"unbiased click through rate: {unbiased_clicks.mean()}"
        )

        return interaction_df

    def _exract_data_by_policy(
        self, interaction_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ランダムポリシーによってログデータを抽出する

        Args:
        - interaction_df (pd.DataFrame): kuairec/small_matrix.csvの
        インタラクションデータ. 評価値行列の密度は100%

        Raises:
            ValueError: behavior_policyがrandomでない場合

        Returns:
        - interaction_df (pd.DataFrame): ランダムポリシーによって生成された
        半人工データ。指定したデータの密度になるようにサンプリングされる。
        """

        if self._params.behavior_policy == "random":
            np.random.seed(self._seed)
            interaction_df = interaction_df.sample(frac=1).reset_index(
                drop=True
            )
            data_size = int(interaction_df.shape[0] * self._params.density)
            interaction_df = interaction_df.iloc[:data_size]
        else:
            self.logger.error(BEHAVIOR_POLICY_NAME_ERROR_MESSAGE)
            raise ValueError(BEHAVIOR_POLICY_NAME_ERROR_MESSAGE)

        return interaction_df

    def _generate_datatype(self, data_size: int) -> list:
        """データの分割設定を元にデータのタイプを生成する

        Args:
        - data_size (int): 抽出したログデータのサイズ

        Returns:
        - (list): ""train", "val", "test"の文字列の配列
        """
        train_val_test_ratio = self._params.train_val_test_ratio
        datatypes = ["train", "val"]

        res = []
        for split_ratio, datatype in zip(train_val_test_ratio, datatypes):
            res += int(split_ratio * data_size) * [datatype]

        res += (data_size - len(res)) * ["test"]

        return res  # shape: (data_size,)

    def _generate_relevance(
        self,
        watch_ratio: pd.Series,
        relevance_clip: float = 2.0,
        normalized_clip: tuple = (0, 1),
    ) -> np.ndarray:
        """半人工的なユーザーとアイテムの関連度を生成する

        Args:
        - watch_ratio (pd.Series): kuairec/small_matrix.csvのインタラクションデータ
        のwatch_ratio. watch_ratio = 動画の視聴時間 / 動画の長さ
        - relevance_clip (float, optional): relevanceの範囲を[0,1]へクリッピングす
        るためのwatch_ratioの閾値. デフォルトは2.0.
        - normalized_clip (tuple, optional): 正規化するための変数.

        Returns:
        - relevance_probabilities  (np.ndarray): 関連度の確率の配列
        """

        relevance_probabilities = np.clip(
            watch_ratio / relevance_clip, *normalized_clip
        )
        return relevance_probabilities

    def _generate_exposure(
        self,
        existing_video_ids: pd.Series,
    ) -> np.ndarray:
        """半人工的なユーザーとアイテムの露出度を生成する。kuairec/big_matrix.csvのインタラクションデータのvideo_idの出現頻度を元に生成する。詳細は、short_paper.mdを参照

        Args:
        - existing_video_ids (pd.Series): 抽出したログデータに存在するvideo_id

        Returns:
        - (np.ndarray): 露出度の確率の配列
        """

        observation_df = KuaiRecCSVLoader.create_big_matrix_df(
            _params=self._params,
            logger=self.logger,
        )

        isin_video_ids = observation_df["video_id"].isin(existing_video_ids)
        video_expo_counts = observation_df[isin_video_ids][
            "video_id"
        ].value_counts()
        del observation_df

        # 標準化してシグモイド関数に通すことで擬似的な露出バイアスを生成
        video_expo_counts = (
            video_expo_counts - video_expo_counts.mean()
        ) / video_expo_counts.std()
        video_exposures = video_expo_counts.apply(_sigmoid)
        exposure_probabilitys = video_exposures[existing_video_ids].values

        return exposure_probabilitys**self._params.exposure_bias

    def _generate_clicks(
        self,
        exposure_probabilities: pd.Series,
        relevance_probabilities: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """半人工的なクリックデータを生成する

        Args:
        - exposure_probabilities (pd.Series): 露出確率のSeries
        - relevance_probabilities (pd.Series): 関連度のSeries

        Returns:
        - biased_clicks (np.ndarray): 露出の影響を被ったクリックデータ
        - relevance_labels (np.ndarray): 露出の影響を被っていないクリックデータ
        """

        # generate clicks
        np.random.seed(self._seed)
        # exposure label ~ Be(P(O = 1))
        exposure_labels = np.random.binomial(
            n=1,
            p=exposure_probabilities,
        )

        # relevance label ~ Be(P(R = 1))
        relevance_labels = np.random.binomial(n=1, p=relevance_probabilities)

        # Y = R * O
        biased_clicks = exposure_labels * relevance_labels

        return biased_clicks, relevance_labels


def _sigmoid(x: float, a: float = 3.0, b: float = -0) -> float:
    """kuairec/big_matrix.csvでのアイテムの出現頻度を確率に変換するためのシグモイド関数

    Args:
        x (float): 標準化されたアイテムの出現頻度

    Returns:
        float: 露出確率
    """
    return 1 / (1 + np.exp(-(a * x + b)))
