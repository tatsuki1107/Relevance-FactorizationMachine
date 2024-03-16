# Standard library imports
from ast import literal_eval
from dataclasses import dataclass
from logging import Logger

# Third-party library imports
from omegaconf import DictConfig, OmegaConf
import pandas as pd

# Internal modules imports
from conf.kuairec import (
    DataFrameConfig,
    VideoDataFrameConfig,
    LogDataPropensityConfig,
)

FILE_NOT_FOUND_ERROR_MESSAGE = (
    "You need user_features.csv, item_category.csv, "
    + "item_daily_features.csv, small_matrix.csv, big_matrix.csv. "
    + "Please install kuairec data at https://kuairec.com/"
    + " and save the data in the ./data/kuairec/ directory."
)

VALUE_ERROR_MESSAGE = (
    "'{}' from '{}'. Please rewrite the conf/config.yaml " + "and conf/config.py."
)


@dataclass
class KuaiRecCSVLoader:
    """load csv files for training and evaluation. see https://kuairec.com/
    or README.md for more details
    """

    @staticmethod
    def create_small_matrix_df(
        _params: DataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """method to load kuairec/small_matrix.csv

        Args:
        - _params (DataFrameConfig): Configuration parameters for the KuaiRec dataset
        - logger (Logger): Logger class instance

        Returns:
            pd.DataFrame: small_matrix.csv data
        """

        feature_names = get_feature_names(_params=_params.used_features)
        usecols = feature_names + ["user_id", "video_id", "watch_ratio"]
        small_matrix_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )

        return small_matrix_df

    @staticmethod
    def create_big_matrix_df(
        _params: LogDataPropensityConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """method to load kuairec/big_matrix.csv

        Args:
        - _params (LogDataPropensityConfig): Configuration parameters for
        the KuaiRec dataset
        - logger (Logger): Logger class instance

        Returns:
            pd.DataFrame: big_matrix.csv data
        """

        usecols = ["user_id", "video_id"]
        big_matrix_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )

        return big_matrix_df

    @staticmethod
    def create_user_features_df(
        existing_user_ids: pd.Series,
        _params: DataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """method to load user_features.csv. FeatureGenerator class extracts only
        the user features of the log data.

        Args:
        - existing_user_ids (pd.Series): user_id of the log data extracted
        by FeatureGenerator
        - _params (DataFrameConfig): Configuration parameters for the KuaiRec dataset
        - logger (Logger): Logger class instance

        Returns:
            pd.DataFrame: user_features.csv data
        """

        feature_names = get_feature_names(_params=_params.used_features)
        usecols = feature_names + ["user_id"]
        user_features_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )
        isin_user_ids = user_features_df["user_id"].isin(existing_user_ids)
        user_features_df = user_features_df[isin_user_ids].reset_index(drop=True)

        return user_features_df

    @staticmethod
    def create_item_features_df(
        existing_video_ids: pd.Series,
        _params: VideoDataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """method to load item_daily_features.csv and item_category.csv.
        FeatureGenerator class extracts only the item features of the log data.

        Args:
        - existing_video_ids (pd.Series): video_id of the log data extracted
        by FeatureGenerator
        - _params (VideoDataFrameConfig): Configuration parameters for
        the KuaiRec dataset
        - logger (Logger): Logger class instance

        Returns:
            pd.DataFrame: item_daily_features.csv and item_category.csv data

        """

        item_daily_features_df = KuaiRecCSVLoader._create_item_daily_features_df(
            existing_video_ids=existing_video_ids,
            _params=_params.daily,
            logger=logger,
        )

        item_categories_df = KuaiRecCSVLoader._create_item_categories_df(
            existing_video_ids=existing_video_ids,
            _params=_params.category,
            logger=logger,
        )
        item_features_df = pd.merge(
            item_daily_features_df, item_categories_df, on="video_id"
        )
        del item_daily_features_df, item_categories_df

        return item_features_df

    @staticmethod
    def _create_item_daily_features_df(
        existing_video_ids: pd.Series,
        _params: DataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """method to load item_daily_features.csv. FeatureGenerator class extracts only
        the item features of the log data.

        Args:
        - existing_video_ids (pd.Series): video_id of the log data extracted
        by FeatureGenerator
        - _params (DataFrameConfig): Configuration parameters for the KuaiRec dataset
        - logger (Logger): Logger class instance

        Returns:
            pd.DataFrame: item_daily_features.csv data
        """

        feature_names = get_feature_names(_params=_params.used_features)
        usecols = feature_names + ["video_id"]
        item_daily_features_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )
        item_daily_features_df = item_daily_features_df.groupby("video_id").first()
        isin_video_ids = item_daily_features_df.index.isin(existing_video_ids)
        item_daily_features_df = item_daily_features_df[isin_video_ids]
        item_daily_features_df.rename_axis("index", inplace=True)
        item_daily_features_df["video_id"] = item_daily_features_df.index.values
        item_daily_features_df.reset_index(inplace=True, drop=True)

        return item_daily_features_df

    @staticmethod
    def _create_item_categories_df(
        existing_video_ids: pd.Series,
        _params: DataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """method to load item_category.csv. FeatureGenerator class extracts only the
        item features of the log data.

        Args:
        - existing_video_ids (pd.Series): video_id of the log data extracted
        by FeatureGenerator
        - _params (DataFrameConfig): Configuration parameters for the KuaiRec dataset
        - logger (Logger): Logger class instance

        Returns:
            pd.DataFrame: item_category.csv data
        """

        feature_names = get_feature_names(_params=_params.used_features)
        usecols = feature_names + ["video_id"]
        item_categories_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )
        isin_video_ids = item_categories_df["video_id"].isin(existing_video_ids)
        item_categories_df = item_categories_df[isin_video_ids].reset_index(drop=True)
        # 文字列から配列へ変換
        item_categories_df["feat"] = item_categories_df["feat"].apply(
            lambda x: literal_eval(x)
        )
        return item_categories_df

    @staticmethod
    def _load_csv(data_path: str, usecols: list, logger: Logger) -> pd.DataFrame:
        """load csv files for training and evaluation

        Args:
        - data_path (str): path to the csv file
        - usecols (list): list of columns to be used
        - logger (Logger): Logger class instance

        Raises:
            ValueError: column name is not found in the csv file
            FileNotFoundError: csv file is not found

        Returns:
            pd.DataFrame: csv file data
        """

        try:
            return pd.read_csv(data_path, usecols=usecols)
        except ValueError as e:
            logger.error(VALUE_ERROR_MESSAGE.format(e, data_path))
            raise
        except FileNotFoundError:
            logger.error(FILE_NOT_FOUND_ERROR_MESSAGE)
            raise


def get_feature_names(_params: DictConfig) -> list:
    """get feature names from the configuration parameters

    Args:
    - _params (DictConfig): Configuration parameters for the KuaiRec dataset

    Returns:
        list: list of feature names
    """

    columns = OmegaConf.to_container(_params)
    return list(columns.keys())
