from omegaconf import DictConfig, OmegaConf
from ast import literal_eval
from dataclasses import dataclass
import pandas as pd
from conf.config import (
    InteractionTableConfig,
    UserTableConfig,
    VideoTableConfig,
    VideoCategoryTableConfig,
    VideoDailyTableConfig,
    LogDataPropensityConfig,
)


@dataclass
class KuaiRecCSVLoader:
    @staticmethod
    def create_interaction_df(params: InteractionTableConfig) -> pd.DataFrame:
        columns = get_features_columns(params=params.features)
        usecols = columns + ["user_id", "video_id", "watch_ratio"]
        interaction_df = KuaiRecCSVLoader._load_csv(
            data_path=params.data_path, usecols=usecols
        )

        return interaction_df

    @staticmethod
    def create_big_matrix_df(params: LogDataPropensityConfig) -> pd.DataFrame:
        usecols = ["user_id", "video_id"]
        observation_df = KuaiRecCSVLoader._load_csv(
            data_path=params.data_path, usecols=usecols
        )

        return observation_df

    @staticmethod
    def create_user_features_df(
        existing_user_ids: pd.Series, params: UserTableConfig
    ) -> pd.DataFrame:
        columns = get_features_columns(params=params.features)
        usecols = columns + ["user_id"]
        user_features_df = KuaiRecCSVLoader._load_csv(
            data_path=params.data_path, usecols=usecols
        )
        isin_user_ids = user_features_df["user_id"].isin(existing_user_ids)
        user_features_df = user_features_df[isin_user_ids].reset_index(
            drop=True
        )

        return user_features_df

    @staticmethod
    def create_item_features_df(
        existing_video_ids: pd.Series, params: VideoTableConfig
    ) -> pd.DataFrame:
        item_daily_features_df = (
            KuaiRecCSVLoader._create_item_daily_features_df(
                existing_video_ids=existing_video_ids, params=params.daily
            )
        )

        item_categories_df = KuaiRecCSVLoader._create_item_categories_df(
            existing_video_ids=existing_video_ids, params=params.category
        )
        item_features_df = pd.merge(
            item_daily_features_df, item_categories_df, on="video_id"
        )
        del item_daily_features_df, item_categories_df

        return item_features_df

    @staticmethod
    def _create_item_daily_features_df(
        existing_video_ids: pd.Series, params: VideoDailyTableConfig
    ) -> pd.DataFrame:
        columns = get_features_columns(params=params.features)
        usecols = columns + ["video_id"]
        item_daily_features_df = KuaiRecCSVLoader._load_csv(
            data_path=params.data_path, usecols=usecols
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
        existing_video_ids: pd.Series, params: VideoCategoryTableConfig
    ) -> pd.DataFrame:
        columns = get_features_columns(params=params.features)
        usecols = columns + ["video_id"]
        item_categories_df = KuaiRecCSVLoader._load_csv(
            data_path=params.data_path, usecols=usecols
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
    def _load_csv(data_path: str, usecols: list) -> pd.DataFrame:
        try:
            df = pd.read_csv(data_path, usecols=usecols)
        except ValueError as e:
            raise ValueError(f"{e} from {data_path}")

        return df


def get_features_columns(params: DictConfig) -> list:
    columns = OmegaConf.to_container(params, resolve=True)
    return list(columns.keys())
