from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from conf.config import LogDataPropensityConfig
from utils.dataloader.base import BaseLoader
from utils.dataloader._kuairec import KuaiRecCSVLoader


@dataclass
class SemiSyntheticLogDataGenerator(BaseLoader):
    _seed: int
    _params: LogDataPropensityConfig

    def load(
        self,
        interaction_df: pd.DataFrame,
    ) -> pd.DataFrame:
        interaction_df = self._exract_data_by_policy(
            interaction_df=interaction_df
        )
        datatypes = self._generate_datatype(
            data_size=interaction_df.shape[0],
        )
        interaction_df["datatype"] = datatypes

        relevance_probabilitys = self._generate_relevance(
            watch_ratio=interaction_df["watch_ratio"]
        )
        interaction_df["relevance"] = relevance_probabilitys
        interaction_df.drop("watch_ratio", axis=1, inplace=True)

        exposure_probabilitys = self._generate_exposure(
            existing_video_ids=interaction_df["video_id"]
        )
        interaction_df["exposure"] = exposure_probabilitys

        biased_clicks, unbiased_clicks = self._generate_clicks(
            exposure_probabilitys=interaction_df["exposure"],
            relevance_probabilitys=interaction_df["relevance"],
        )
        interaction_df["biased_click"] = biased_clicks
        interaction_df["unbiased_click"] = unbiased_clicks

        return interaction_df

    def _exract_data_by_policy(
        self, interaction_df: pd.DataFrame
    ) -> pd.DataFrame:
        # 過去の推薦方策pi_bはランダムなポリシーとしてログデータを生成
        # ユーザの評価は時間に左右されないと仮定
        if self._params.behavior_policy == "random":
            np.random.seed(self._seed)
            interaction_df = interaction_df.sample(frac=1).reset_index(
                drop=True
            )
            data_size = int(interaction_df.shape[0] * self._params.density)
            interaction_df = interaction_df.iloc[:data_size]
        else:
            raise ValueError("behavior_policy must be random")

        return interaction_df

    def _generate_datatype(self, data_size: int) -> list:
        train_val_test_ratio = self._params.train_val_test_ratio
        datatypes = ["train", "val"]

        res = []
        for split_ratio, datatype in zip(train_val_test_ratio, datatypes):
            res += int(split_ratio * data_size) * [datatype]

        res += (data_size - len(res)) * ["test"]

        return res

    def _generate_relevance(
        self,
        watch_ratio: pd.Series,
        relevance_clip: float = 2.0,
        normalized_clip: tuple = (0, 1),
    ) -> np.ndarray:
        # watch ratio >= 2を1とした基準で関連度を生成
        relevance_probabilitys = np.clip(
            watch_ratio / relevance_clip, *normalized_clip
        )
        return relevance_probabilitys

    def _generate_exposure(
        self,
        existing_video_ids: pd.Series,
    ) -> pd.DataFrame:
        observation_df = KuaiRecCSVLoader.create_big_matrix_df(
            _params=self._params
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
        exposure_probabilitys: pd.Series,
        relevance_probabilitys: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # generate clicks
        np.random.seed(self._seed)
        # P(O = 1)
        exposure_labels = np.random.binomial(
            n=1,
            p=exposure_probabilitys,
        )

        # P(R = 1)
        relevance_labels = np.random.binomial(n=1, p=relevance_probabilitys)

        # P(Y = 1) = P(R = 1) * P(O = 1)
        biased_clicks = exposure_labels * relevance_labels

        return biased_clicks, relevance_labels


def _sigmoid(x: float, a: float = 3.0, b: float = -0) -> float:
    return 1 / (1 + np.exp(-(a * x + b)))
