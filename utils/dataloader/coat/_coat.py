# Standard library imports
import codecs
from dataclasses import dataclass
from typing import Tuple
from logging import Logger

# Third-party library imports
import numpy as np
import pandas as pd

# Internal modules imports
from utils.dataloader.base import BaseLoader
from conf.coat import TableConfig

PROPENSITY_SCORE_MESSAGE = "pscore distribution mean: {}, std: {}, min: {}, max: {}"

FILE_NOT_FOUND_ERROR_MESSAGE = (
    "You need 1. train.ascii, 2. test.ascii, 3. propensities.ascii, "
    "4. user_features(.txt .ascii), and 5. item_features(.txt .ascii) files"
    "to load the Coat dataset."
    "Please install kuairec data at https://www.cs.cornell.edu/~schnabts/mnar/"
    " and save the data in the ./data/coat/ directory."
)


@dataclass
class CoatCSVLoader(BaseLoader):
    """Load the Coat dataset. see https://www.cs.cornell.edu/~schnabts/mnar/
    for more details.

    Args:
    - params (TableConfig): Configuration parameters for the Coat dataset (read-only)
    - seed (int): Random seed (read-only)
    - logger (Logger): Logger class instance
    """

    _params: TableConfig
    _seed: int
    logger: Logger

    def load(self) -> Tuple[pd.DataFrame, ...]:
        """Load the Coat dataset

        Returns:
            Tuple[pd.DataFrame, ...]: Train data, test data, user features,
            and item features
        """
        pscore_df = self._create_pscore_df()
        train_df, test_df = self._create_interaction(pscore_df)
        user_features_df, item_features_df = self._create_features_df()

        return train_df, test_df, user_features_df, item_features_df

    def _create_interaction(
        self, pscore_df: pd.DataFrame, rate_threshold: int = 4
    ) -> Tuple[pd.DataFrame, ...]:
        """Create the interaction data

        Args:
            pscore_df (pd.DataFrame): Propensity score dataframe

        Returns:
            Tuple[pd.DataFrame, ...]: Train data and test data
        """

        # train data
        train_df = self._load_csv(path=self._params.train.data_path)
        converted_cols = {"level_0": "user", "level_1": "item", 0: "rate"}
        train_df = train_df.stack().reset_index().rename(columns=converted_cols)
        train_df = train_df[train_df["rate"] != 0].reset_index(drop=True)
        train_df["rate"] = transform_rating(train_df["rate"])
        np.random.seed(self._seed)
        train_df["label"] = np.random.binomial(p=train_df["rate"], n=1)
        train_df.drop("rate", axis=1, inplace=True)

        # test data
        test_df = self._load_csv(path=self._params.test.data_path)
        converted_cols = {"level_0": "user", "level_1": "item", 0: "label"}
        test_df = test_df.stack().reset_index().rename(columns=converted_cols)
        test_df = test_df[test_df["label"] != 0].reset_index(drop=True)
        test_df["label"] = (test_df["label"] >= rate_threshold).astype(int)

        train_df = pd.merge(train_df, pscore_df, on=["item"], how="left")
        test_df = pd.merge(test_df, pscore_df, on=["item"], how="left")

        return train_df, test_df

    def _create_pscore_df(self) -> pd.DataFrame:
        """Create the propensity score dataframe

        Returns:
            pd.DataFrame: Propensity score dataframe
        """

        _df = self._load_csv(path=self._params.propensities.data_path)
        cols = {"level_0": "user", "level_1": "item", 0: "pscore"}
        _df = _df.stack().reset_index().rename(columns=cols)

        pscores = _df.groupby("item").agg({"pscore": "mean"}).values.reshape(-1)
        pscore_df = pd.DataFrame()
        pscore_df["item"] = np.arange(_df["item"].max() + 1)
        pscore_df["pscore"] = pscores
        pscore_df["ones_pscore"] = 1.0

        pscore_df["pscore"] = pscore_df["pscore"] / pscore_df["pscore"].max()

        self.logger.info(
            PROPENSITY_SCORE_MESSAGE.format(
                pscore_df["pscore"].mean(),
                pscore_df["pscore"].std(),
                pscore_df["pscore"].min(),
                pscore_df["pscore"].max(),
            )
        )

        return pscore_df

    def _create_features_df(self) -> Tuple[pd.DataFrame, ...]:
        """Create the user and item features dataframe

        Returns:
            Tuple[pd.DataFrame, ...]: User features and item features dataframe
        """

        conf = {"user": self._params.user_features, "item": self._params.item_features}

        feature_df_dict = dict()
        for resource in ["user", "item"]:
            cols = self._load_txt(path=conf[resource].txt_path)
            cols = cols.split("\n")

            feature_cols = {}
            for i in range(len(cols)):
                feature_cols[i] = cols[i]

            features_df = self._load_csv(path=conf[resource].data_path)
            features_df = features_df.rename(columns=feature_cols)

            feature_df_dict[resource] = features_df

        user_features_df: pd.DataFrame = feature_df_dict["user"]
        item_features_df: pd.DataFrame = feature_df_dict["item"]

        return user_features_df, item_features_df

    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load the csv file

        Args:
        - path (str): File path

        Returns:
            pd.DataFrame: Dataframe
        """

        try:
            with codecs.open(path, "r", "utf-8", errors="ignore") as f:
                return pd.read_csv(f, delimiter=" ", header=None)
        except FileNotFoundError:
            self.logger.error(FILE_NOT_FOUND_ERROR_MESSAGE)
            raise

    def _load_txt(self, path: str) -> str:
        """Load the txt file

        Args:
        - path (str): File path

        Returns:
            str: Text
        """

        try:
            with open(path, "r") as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(FILE_NOT_FOUND_ERROR_MESSAGE)
            raise


def transform_rating(ratings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Transform ratings into graded relevance information.

    Args:
    - ratings (np.ndarray): Ratings
    - eps (float): Epsilon

    Returns:
    - np.ndarray: Graded relevance information

    """
    ratings -= 1
    return eps + (1.0 - eps) * (2**ratings - 1) / (2 ** np.max(ratings) - 1)
