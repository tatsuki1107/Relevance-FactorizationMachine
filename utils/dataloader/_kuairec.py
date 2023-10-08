from omegaconf import DictConfig, OmegaConf
from ast import literal_eval
from dataclasses import dataclass
from logging import Logger
import pandas as pd
from conf.config import (
    InteractionTableConfig,
    UserTableConfig,
    VideoTableConfig,
    VideoCategoryTableConfig,
    VideoDailyTableConfig,
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
    @staticmethod
    def create_interaction_df(
        _params: InteractionTableConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        columns = get_features_columns(_params=_params.used_features)
        usecols = columns + ["user_id", "video_id", "watch_ratio"]
        interaction_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )

        return interaction_df

    @staticmethod
    def create_big_matrix_df(
        _params: LogDataPropensityConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        usecols = ["user_id", "video_id"]
        observation_df = KuaiRecCSVLoader._load_csv(
            data_path=_params.data_path,
            usecols=usecols,
            logger=logger,
        )

        return observation_df

    @staticmethod
    def create_user_features_df(
        existing_user_ids: pd.Series,
        _params: UserTableConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        columns = get_features_columns(_params=_params.used_features)
        usecols = columns + ["user_id"]
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
        _params: VideoTableConfig,
        logger: Logger,
    ) -> pd.DataFrame:
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
        _params: VideoDailyTableConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        columns = get_features_columns(_params=_params.used_features)
        usecols = columns + ["video_id"]
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
        _params: VideoCategoryTableConfig,
        logger: Logger,
    ) -> pd.DataFrame:
        columns = get_features_columns(_params=_params.used_features)
        usecols = columns + ["video_id"]
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
        try:
            df = pd.read_csv(data_path, usecols=usecols)
        except ValueError as e:
            logger.error(VALUE_ERROR_MESSAGE.format(e, data_path))
            raise ValueError(VALUE_ERROR_MESSAGE.format(e, data_path))
        except FileNotFoundError:
            logger.error(FILE_NOT_FOUND_ERROR_MESSAGE)
            raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

        return df


def get_features_columns(_params: DictConfig) -> list:
    columns = OmegaConf.to_container(_params, resolve=True)
    return list(columns.keys())
