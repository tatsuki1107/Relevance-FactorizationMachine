# Standard library imports
from ast import literal_eval
from dataclasses import dataclass
from logging import Logger

# Third-party library imports
from omegaconf import DictConfig, OmegaConf
import pandas as pd

# Internal modules imports
from conf.config import (
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
    "'{}' from '{}'. Please rewrite the conf/config.yaml "
    + "and conf/config.py."
)


@dataclass
class KuaiRecCSVLoader:
    """KuaiRecのCSVファイルを読み込むクラス.
    こちらのデータの詳細は, https://kuairec.com/ ,またはREADME.mdを参照してください.
    """

    @staticmethod
    def create_small_matrix_df(
        _params: DataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """small_matrix.csvを読み込むメソッド

        Args:
        - _params (InteractionTableConfig): インタラクションの設定パラメータ (read only)
        - logger (Logger): Loggerクラスのインスタンス

        Returns:
        - small_matrix_df (pd.DataFrame): small_matrix.csvのデータ
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
        """big_matrix.csvを読み込むメソッド

        Args:
        - _params (LogDataPropensityConfig): 半人工データ生成の設定パラメータ (read only)
        - logger (Logger): Loggerクラスのインスタンス

        Returns:
        - big_matrix_df (pd.DataFrame): big_matrix.csvのデータ
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
        """user_features.csvを読み込むメソッド.FeatureGeneratorクラスで抽出したログデータに存在するユーザーの特徴量のみを抽出する.

        Args:
        - existing_user_ids (pd.Series): 抽出したログデータに存在するユーザーID
        - _params (UserTableConfig): ユーザーの特徴量の設定パラメータ (read only)
        - logger (Logger): Loggerクラスのインスタンス

        Returns:
        - user_features_df (pd.DataFrame): user_features.csvのデータ
        """
        feature_names = get_feature_names(_params=_params.used_features)
        usecols = feature_names + ["user_id"]
        user_features_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )
        isin_user_ids = user_features_df["user_id"].isin(existing_user_ids)
        user_features_df = user_features_df[isin_user_ids].reset_index(
            drop=True
        )

        return user_features_df

    @staticmethod
    def create_item_features_df(
        existing_video_ids: pd.Series,
        _params: VideoDataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """item_category.csvとitem_daily_features.csvを統合したcsvファイルを読み込むメソッド.

        Args:
        - existing_video_ids (pd.Series): 抽出したログデータに存在する動画ID
        - _params (VideoTableConfig): 動画の特徴量の設定パラメータ (read only)
        - logger (Logger): Loggerクラスのインスタンス

        Returns:
        - item_features_df (pd.DataFrame): item_category.csvと
        item_daily_features.csvを統合したデータ
        """

        item_daily_features_df = (
            KuaiRecCSVLoader._create_item_daily_features_df(
                existing_video_ids=existing_video_ids,
                _params=_params.daily,
                logger=logger,
            )
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
        """item_daily_features.csvを読み込むメソッド.FeatureGeneratorクラスで抽出したログデータに存在する動画の特徴量のみを抽出する.動画ごとにtimestampで降順ソートし、最初の行のみを抽出する.

        Args:
        - existing_video_ids (pd.Series): 抽出したログデータに存在する動画ID
        - _params (VideoDailyTableConfig): dailyの動画特徴量の設定パラメータ (read only)
        - logger (Logger): Loggerクラスのインスタンス

        Returns:
        - item_daily_features_df (pd.DataFrame): item_daily_features.csvのデータ
        """

        feature_names = get_feature_names(_params=_params.used_features)
        usecols = feature_names + ["video_id"]
        item_daily_features_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )
        item_daily_features_df = item_daily_features_df.groupby(
            "video_id"
        ).first()
        isin_video_ids = item_daily_features_df.index.isin(existing_video_ids)
        item_daily_features_df = item_daily_features_df[isin_video_ids]
        item_daily_features_df.rename_axis("index", inplace=True)
        item_daily_features_df[
            "video_id"
        ] = item_daily_features_df.index.values
        item_daily_features_df.reset_index(inplace=True, drop=True)

        return item_daily_features_df

    @staticmethod
    def _create_item_categories_df(
        existing_video_ids: pd.Series,
        _params: DataFrameConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        """item_category.csvを読み込むメソッド.FeatureGeneratorクラスで抽出したログデータに存在する動画の特徴量のみを抽出する.

        Args:
        - existing_video_ids (pd.Series): 抽出したログデータに存在する動画ID
        - _params (VideoCategoryTableConfig): categoryの動画特徴量の
        設定パラメータ (read only)
        - logger (Logger): Loggerクラスのインスタンス

        Returns:
        - item_categories_df (pd.DataFrame): item_category.csvのデータ
        """
        feature_names = get_feature_names(_params=_params.used_features)
        usecols = feature_names + ["video_id"]
        item_categories_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )
        isin_video_ids = item_categories_df["video_id"].isin(
            existing_video_ids
        )
        item_categories_df = item_categories_df[isin_video_ids].reset_index(
            drop=True
        )
        # 文字列から配列へ変換
        item_categories_df["feat"] = item_categories_df["feat"].apply(
            lambda x: literal_eval(x)
        )
        return item_categories_df

    @staticmethod
    def _load_csv(
        data_path: str, usecols: list, logger: Logger
    ) -> pd.DataFrame:
        """CSVファイルを読み込むメソッド

        Args:
        - data_path (str): CSVファイルのパス
        - usecols (list): 読み込むカラム名のリスト
        - logger (Logger): Loggerクラスのインスタンス

        Raises:
        - ValueError: 指定したカラム名が特定のcsvに存在しない場合
        - FileNotFoundError: 指定したパスにファイルが存在しない場合

        Returns:
        - df (pd.DataFrame): CSVファイルのデータ
        """
        try:
            df = pd.read_csv(data_path, usecols=usecols)
        except ValueError as e:
            logger.error(VALUE_ERROR_MESSAGE.format(e, data_path))
            raise ValueError(VALUE_ERROR_MESSAGE.format(e, data_path))
        except FileNotFoundError:
            logger.error(FILE_NOT_FOUND_ERROR_MESSAGE)
            raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

        return df


def get_feature_names(_params: DictConfig) -> list:
    """hydraで指定した特徴量のカラム名をlistとして取得する

    Args:
    - _params (DictConfig): hydraで指定した特徴量のカラム名

    Returns:
    - (list): 特徴量のカラム名
    """

    columns = OmegaConf.to_container(_params)
    return list(columns.keys())
