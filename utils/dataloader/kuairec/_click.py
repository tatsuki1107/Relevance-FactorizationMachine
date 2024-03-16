# Standard library imports
from dataclasses import dataclass
from logging import Logger
from typing import Tuple

# Third-party library imports
import numpy as np
import pandas as pd

# Internal modules imports
from conf.kuairec import LogDataPropensityConfig
from utils.dataloader.base import BaseLoader
from utils.dataloader.kuairec._kuairec import KuaiRecCSVLoader


BEHAVIOR_POLICY_NAME_ERROR_MESSAGE = "behavior_policy must be random."
REPRESENTATIVE_VALUE_MESSAGE = "{} distribution mean: {}, std: {}, min: {}, max: {}"
CLICK_THROUGH_RATE_MESSAGE = (
    "biased click through rate: {}, " + "unbiased click through rate: {}"
)


@dataclass
class SemiSyntheticLogDataGenerator(BaseLoader):
    """generate semi-synthetic log data for training and evaluation

    Args:
    - _seed (int): random seed
    - _params (LogDataPropensityConfig): Configuration parameters for
    the KuaiRec dataset (read-only)
    - logger (Logger): Logger class instance
    """

    _seed: int
    _params: LogDataPropensityConfig
    logger: Logger

    def load(
        self,
        interaction_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """generate semi-synthetic log data for training and evaluation

        Args:
        - interaction_df (pd.DataFrame): interaction data in kuairec/small_matrix.csv

        Returns:
            pd.DataFrame: semi-synthetic log data for training and evaluation
        """

        interaction_df = self._exract_data_by_policy(interaction_df=interaction_df)
        data_size = interaction_df.shape[0]
        self.logger.info(f"log data size: {data_size}")
        datatypes = self._generate_datatype(
            data_size=data_size,
        )
        interaction_df["datatype"] = datatypes

        relevance_probabilities = self._generate_relevance(
            watch_ratio=interaction_df["watch_ratio"]
        )
        interaction_df["gamma"] = relevance_probabilities
        interaction_df.drop("watch_ratio", axis=1, inplace=True)
        self.logger.info(
            REPRESENTATIVE_VALUE_MESSAGE.format(
                "gamma",
                relevance_probabilities.mean(),
                relevance_probabilities.std(),
                relevance_probabilities.min(),
                relevance_probabilities.max(),
            )
        )

        exposure_probabilities = self._generate_exposure(
            existing_video_ids=interaction_df["video_id"]
        )
        interaction_df["pscore"] = exposure_probabilities
        interaction_df["ones_pscore"] = 1.0
        self.logger.info(
            REPRESENTATIVE_VALUE_MESSAGE.format(
                "pscore",
                exposure_probabilities.mean(),
                exposure_probabilities.std(),
                exposure_probabilities.min(),
                exposure_probabilities.max(),
            )
        )

        biased_clicks, unbiased_clicks = self._generate_clicks(
            exposure_probabilities=interaction_df["pscore"],
            relevance_probabilities=interaction_df["gamma"],
        )
        interaction_df["label"] = biased_clicks
        interaction_df["relevance"] = unbiased_clicks
        self.logger.info(
            CLICK_THROUGH_RATE_MESSAGE.format(
                biased_clicks.mean(), unbiased_clicks.mean()
            )
        )

        return interaction_df

    def _exract_data_by_policy(self, interaction_df: pd.DataFrame) -> pd.DataFrame:
        """Extract data by random policy

        Args:
        - interaction_df (pd.DataFrame): interaction data in kuairec/small_matrix.csv

        Raises:
            ValueError: behavior_policy must be random.

        Returns:
            pd.DataFrame: extracted interaction data
        """

        if self._params.behavior_policy == "random":
            np.random.seed(self._seed)
            interaction_df = interaction_df.sample(frac=1).reset_index(drop=True)
            data_size = int(interaction_df.shape[0] * self._params.density)
            interaction_df = interaction_df.iloc[:data_size]
        else:
            self.logger.error(BEHAVIOR_POLICY_NAME_ERROR_MESSAGE)
            raise ValueError(BEHAVIOR_POLICY_NAME_ERROR_MESSAGE)

        return interaction_df

    def _generate_datatype(self, data_size: int) -> list:
        """Generate datatype for training, validation, and test

        Args:
        - data_size (int): size of the interaction data

        Returns:
            list: datatype for training, validation, and test
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
        """Generate relevance probabilities

        Args:
        - watch_ratio (pd.Series): watch ratio of the video
        - relevance_clip (float, optional): clip value for relevance. Defaults to 2.0.
        - normalized_clip (tuple, optional): scale for relevance. Defaults to (0, 1).

        Returns:
            np.ndarray: _description_
        """

        relevance_probabilities = np.clip(
            watch_ratio / relevance_clip, *normalized_clip
        )
        return relevance_probabilities

    def _generate_exposure(
        self, existing_video_ids: pd.Series, eps: float = 0.1
    ) -> np.ndarray:
        """Generate exposure probabilities based on the frequency of the video in the
        kuairec/big_matrix.csv

        Args:
        - existing_video_ids (pd.Series): video ids in the interaction data
        - eps (float, optional): small value to avoid zero division. Defaults to 0.1.

        Returns:
            np.ndarray: exposure probabilities
        """

        observation_df = KuaiRecCSVLoader.create_big_matrix_df(
            _params=self._params,
            logger=self.logger,
        )

        isin_video_ids = observation_df["video_id"].isin(existing_video_ids)
        video_expo_counts = observation_df[isin_video_ids]["video_id"].value_counts()
        del observation_df

        # generate exposure probabilities using sigmoid function
        video_expo_counts = (
            video_expo_counts - video_expo_counts.mean()
        ) / video_expo_counts.std()
        video_exposures = video_expo_counts.apply(_sigmoid)

        exposure_probabilitys = (
            video_exposures[existing_video_ids].values ** self._params.exposure_bias
        )
        exposure_probabilitys = np.maximum(exposure_probabilitys, eps)

        return exposure_probabilitys

    def _generate_clicks(
        self,
        exposure_probabilities: pd.Series,
        relevance_probabilities: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate biased clicks and unbiased clicks

        Args:
        - exposure_probabilities (pd.Series): exposure probabilities
        - relevance_probabilities (pd.Series): relevance probabilities

        Returns:
            Tuple[np.ndarray, np.ndarray]: biased clicks and unbiased clicks
        """

        # generate clicks
        np.random.seed(self._seed)
        # exposure label(O_{u,i}) ~ Be(\theta_{u,i})
        exposure_labels = np.random.binomial(
            n=1,
            p=exposure_probabilities,
        )

        # relevance label(R_{u,i}) ~ Be(\gamma_{u,i})
        relevance_labels = np.random.binomial(n=1, p=relevance_probabilities)

        # Y = R * O
        biased_clicks = exposure_labels * relevance_labels

        return biased_clicks, relevance_labels


def _sigmoid(x: float, a: float = 3.0, b: float = -1.0) -> float:
    """Sigmoid function"""
    return 1 / (1 + np.exp(-(a * x + b)))
