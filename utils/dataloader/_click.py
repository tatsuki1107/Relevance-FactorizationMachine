from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from conf.config import LogDataPropensityConfig
from utils.dataloader._kuairec import KuaiRecCSVLoader


@dataclass
class ClickDataGenerator:
    seed: int

    def generate_logdata_using_observed_data(
        self,
        interaction_df: pd.DataFrame,
        params: LogDataPropensityConfig,
    ) -> pd.DataFrame:
        interaction_df = self._exract_data_by_policy(
            interaction_df=interaction_df, params=params
        )

        relevance_probabilitys = self._generate_relevance(
            interaction_df=interaction_df[["watch_ratio"]]
        )
        interaction_df["relevance"] = relevance_probabilitys
        interaction_df.drop("watch_ratio", axis=1, inplace=True)

        exposure_probabilitys = self._generate_exposure(
            interaction_df=interaction_df[["user_id", "video_id"]],
            params=params,
        )
        interaction_df["exposure"] = exposure_probabilitys

        biased_clicks, unbiased_clicks = self._generate_clicks(
            interaction_df=interaction_df[["relevance", "exposure"]]
        )
        interaction_df["biased_click"] = biased_clicks
        interaction_df["unbiased_click"] = unbiased_clicks

        return interaction_df

    def _exract_data_by_policy(
        self, interaction_df: pd.DataFrame, params: LogDataPropensityConfig
    ) -> pd.DataFrame:
        # 過去の推薦方策pi_bはランダムなポリシーとしてログデータを生成
        # ユーザの評価は時間に左右されないと仮定
        if params.behavior_policy == "random":
            np.random.seed(self.seed)
            interaction_df = interaction_df.sample(frac=1).reset_index(
                drop=True
            )
            data_size = int(interaction_df.shape[0] * params.density)
            interaction_df = interaction_df.iloc[:data_size]
        else:
            raise ValueError("behavior_policy must be random")

        return interaction_df

    def _generate_relevance(
        self, interaction_df: pd.DataFrame, normalized_clip: tuple = (0, 1)
    ) -> np.ndarray:
        # watch ratio >= 2を1とした基準で関連度を生成
        watch_ratio = interaction_df["watch_ratio"].values
        relevance_probabilitys = np.clip(watch_ratio / 2, *normalized_clip)
        return relevance_probabilitys

    def _generate_exposure(
        self, interaction_df: pd.DataFrame, params: LogDataPropensityConfig
    ) -> pd.DataFrame:
        observation_df = KuaiRecCSVLoader.create_big_matrix_df(params=params)

        existing_video_ids = interaction_df["video_id"]
        isin_video_ids = observation_df["video_id"].isin(existing_video_ids)
        video_expo_counts = observation_df[isin_video_ids][
            "video_id"
        ].value_counts()

        existing_user_ids = interaction_df["user_id"]
        isin_user_ids = observation_df["user_id"].isin(existing_user_ids)
        user_expo_counts = observation_df[isin_user_ids][
            "user_id"
        ].value_counts()
        del observation_df

        video_expo_probs = (
            video_expo_counts / video_expo_counts.max()
        ) ** params.exposure_bias
        user_expo_probs = (
            user_expo_counts / user_expo_counts.max()
        ) ** params.exposure_bias
        exposure_probabilitys = video_expo_probs[existing_video_ids].values
        exposure_probabilitys *= user_expo_probs[existing_user_ids].values

        return exposure_probabilitys

    def _generate_clicks(
        self, interaction_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        # generate clicks
        np.random.seed(self.seed)
        # バイアスのっかかったクリックデータを生成   P(Y = 1) = P(R = 1) * P(O = 1)
        biased_click = np.random.binomial(
            n=1,
            p=interaction_df["relevance"] * interaction_df["exposure"],
        )

        # テストデータ用のクリックデータ P(Y = 1) = P(R = 1)
        unbiased_click = np.random.binomial(n=1, p=interaction_df["relevance"])

        return biased_click, unbiased_click
