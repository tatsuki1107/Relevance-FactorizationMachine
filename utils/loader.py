from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, Optional
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from ast import literal_eval
from conf.config import (
    ExperimentConfig,
    InteractionTableConfig,
    LogDataPropensityConfig,
    UserTableConfig,
    VideoDailyTableConfig,
    VideoCategoryTableConfig,
    VideoTableConfig,
)
from utils.model import FMDataset, PMFDataset


@dataclass
class DataLoader:
    features: Optional[csr_matrix] = None
    interaction_df: Optional[pd.DataFrame] = None

    def load(self, params: ExperimentConfig) -> tuple:
        """半人工データを生成する"""

        # small_matrix.csvのインタラクションデータを研究に用いる
        self._create_interaction_df(params=params.tables.interaction)
        # 自然に観測されたbig_matrix上でのユーザーとアイテムの相対的な露出を用いて、
        # クリックデータを生成する
        self._generate_logdata_using_observed_data(
            params=params.logdata_propensity, seed=params.seed
        )
        # interaction_dfに存在するユーザーとアイテムの特徴量を生成する
        self._create_features(params=params)
        # FMとPMFを用いて学習、評価できるようにデータを整形する
        datasets = self._prepare_datasets(params=params.logdata_propensity)
        return datasets

    def _create_interaction_df(self, params: InteractionTableConfig) -> None:
        columns = self._get_features_columns(params=params.features)
        usecols = columns + ["user_id", "video_id", "watch_ratio"]
        self.interaction_df = pd.read_csv(params.data_path, usecols=usecols)

    def _create_features(self, params: ExperimentConfig) -> None:
        existing_unique_user_ids = self.interaction_df["user_id"].unique()
        existing_unique_video_ids = self.interaction_df["video_id"].unique()

        user_id2_index = {}
        for i, user_id in enumerate(np.sort(existing_unique_user_ids)):
            user_id2_index[user_id] = i

        # 観測されたユーザIDを0から順に振り直す
        self.interaction_df["user_index"] = self.interaction_df[
            "user_id"
        ].apply(lambda x: user_id2_index[x])

        video_id2_index = {}
        for i, video_id in enumerate(np.sort(existing_unique_video_ids)):
            video_id2_index[video_id] = i

        # 観測された動画IDを0から順に振り直す
        self.interaction_df["video_index"] = self.interaction_df[
            "video_id"
        ].apply(lambda x: video_id2_index[x])

        sparse_user_indices = csr_matrix(
            pd.get_dummies(
                self.interaction_df["user_index"], drop_first=True, dtype=int
            ).values
        )
        sparse_video_indices = csr_matrix(
            pd.get_dummies(
                self.interaction_df["video_index"], drop_first=True, dtype=int
            ).values
        )
        basefeatures = hstack([sparse_user_indices, sparse_video_indices])
        del sparse_user_indices, sparse_video_indices

        user_features_df = self._create_user_features_df(
            existing_user_ids=self.interaction_df["user_id"],
            params=params.tables.user,
        )
        item_features_df = self._create_item_features_df(
            existing_video_ids=self.interaction_df["video_id"],
            params=params.tables.video,
        )

        dataframes = {
            "interaction": self.interaction_df,
            "user": user_features_df,
            "video": item_features_df,
        }

        tables_dict = OmegaConf.to_container(params.tables, resolve=True)

        features = [basefeatures]
        for df_name, df in dataframes.items():
            tables = tables_dict[df_name]

            if df_name == "video":
                columns = (
                    tables["daily"]["features"]
                    | tables["category"]["features"]
                )
            else:
                columns = tables["features"]

            converted_df = self._feature_engineering(df=df, columns=columns)
            if df_name == "interaction":
                columns = list(columns.keys())
                sparse_features = csr_matrix(converted_df[columns].values)

            else:
                id = f"{df_name}_id"
                features_df = pd.merge(
                    self.interaction_df[[id]],
                    converted_df,
                    on=id,
                    how="left",
                )
                features_df.drop([id], axis=1, inplace=True)
                sparse_features = csr_matrix(features_df.values)

            features.append(sparse_features)

        self.features = hstack(features)
        self.interaction_df.drop(["user_id", "video_id"], axis=1, inplace=True)
        del user_features_df, item_features_df

    def _prepare_datasets(self, params: LogDataPropensityConfig) -> dict:
        if self.features is None:
            raise ValueError(
                "You must create features before preparing datasets"
            )

        # split train, val, test
        data_size = self.interaction_df.shape[0]
        split_index = [
            int(data_size * ratio) for ratio in params.train_val_test_ratio[:2]
        ]
        split_index[1] += split_index[0]
        train_indices = np.arange(split_index[0])
        val_indices = np.arange(split_index[0], split_index[1])
        test_indices = np.arange(split_index[1], data_size)

        dataset_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        # prepare fm_datasets
        fm_datasets = self._prepare_fm_datasets(
            dataset_indices=dataset_indices
        )

        # prepare pmf_datasets
        pmf_datasets = self._prarpare_pmf_datasets(
            dataset_indices=dataset_indices
        )

        # prepare pscores and clicks
        pscores, clicks = self._prepare_pscores_and_clicks(
            dataset_indices=dataset_indices,
        )

        # prepare test_user2indices
        test_user2indices = self._create_test_user2indices(
            test_indices=test_indices,
        )

        datasets = {
            "FM": fm_datasets,
            "PMF": pmf_datasets,
            "clicks": clicks,
            "pscores": pscores,
            "test_user2indices": test_user2indices,
        }

        return datasets

    def _prepare_fm_datasets(
        self,
        dataset_indices: dict,
    ) -> FMDataset:
        datasets = {}
        for _datatype, _indices in dataset_indices.items():
            datasets[_datatype] = self.features[_indices]

        fm_datasets = FMDataset(**datasets)

        return fm_datasets

    def _prarpare_pmf_datasets(
        self,
        dataset_indices: dict,
    ) -> PMFDataset:
        datasets = {}
        columns = ["user_index", "video_index"]
        for _datatype, _indices in dataset_indices.items():
            datasets[_datatype] = self.interaction_df.iloc[_indices][
                columns
            ].values

        n_users = self.interaction_df["user_index"].max() + 1
        n_items = self.interaction_df["video_index"].max() + 1
        datasets["n_users"] = n_users
        datasets["n_items"] = n_items

        pmf_datasets = PMFDataset(**datasets)

        return pmf_datasets

    def _prepare_pscores_and_clicks(
        self,
        dataset_indices: dict,
    ) -> Tuple[dict, defaultdict]:
        pscores, clicks = {}, defaultdict(dict)
        for _data, _indices in dataset_indices.items():
            if _data in {"train", "val"}:
                pscores[_data] = self.interaction_df.iloc[_indices][
                    "exposure"
                ].values
                clicks[_data]["biased"] = self.interaction_df.iloc[_indices][
                    "biased_click"
                ].values

            clicks[_data]["unbiased"] = self.interaction_df.iloc[_indices][
                "unbiased_click"
            ].values

        return pscores, clicks

    def _create_test_user2indices(
        self,
        test_indices: np.ndarray,
        thetahold: int = 0.3,
    ) -> dict:
        # all
        test_df = self.interaction_df.iloc[test_indices].reset_index(drop=True)

        filter = test_df["exposure"] <= thetahold
        # rare
        rare_test_df = test_df[filter]

        # popular
        popular_test_df = test_df[~filter]

        dataframes = {
            "all": test_df,
            "rare": rare_test_df,
            "popular": popular_test_df,
        }
        test_user2indices = {}
        for frequency, df in dataframes.items():
            groups = df.sort_values(
                by=["user_index", "unbiased_click"], ascending=[True, False]
            ).groupby("user_index")
            df_indices_per_user = []
            for _, group in groups:
                df_indices_per_user.append(group.index.tolist())

            test_user2indices[frequency] = df_indices_per_user

        del test_df, rare_test_df, popular_test_df

        return test_user2indices

    def _feature_engineering(
        self, df: pd.DataFrame, columns: dict
    ) -> pd.DataFrame:
        datatypes = defaultdict(list)
        for feature_name, datatype in columns.items():
            datatypes[datatype].append(feature_name)

        if datatypes["label"]:
            df = pd.get_dummies(
                df, columns=datatypes["label"], drop_first=True, dtype=int
            )

        if datatypes["int"] or datatypes["float"]:
            columns = datatypes["int"] + datatypes["float"]
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            # 欠損値はひとまず平均値で埋める
            df[columns] = df[columns].fillna(df[columns].mean())

        if datatypes["multilabel"]:
            for column in datatypes["multilabel"]:
                multilabels = df[column].to_list()
                mlb = MultiLabelBinarizer()
                multi_hot_tags = mlb.fit_transform(multilabels)
                multi_hot_df = pd.DataFrame(multi_hot_tags)
                df = pd.concat([df, multi_hot_df], axis=1)
                df.drop(column, axis=1, inplace=True)

        return df

    def _create_user_features_df(
        self, existing_user_ids: pd.Series, params: UserTableConfig
    ) -> pd.DataFrame:
        columns = self._get_features_columns(params=params.features)
        usecols = columns + ["user_id"]
        user_features_df = pd.read_csv(params.data_path, usecols=usecols)
        isin_user_ids = user_features_df["user_id"].isin(existing_user_ids)
        user_features_df = user_features_df[isin_user_ids].reset_index(
            drop=True
        )

        return user_features_df

    def _create_item_features_df(
        self, existing_video_ids: pd.Series, params: VideoTableConfig
    ) -> pd.DataFrame:
        item_daily_features_df = self._create_item_daily_features_df(
            existing_video_ids=existing_video_ids, params=params.daily
        )

        item_categories_df = self._create_item_categories_df(
            existing_video_ids=existing_video_ids, params=params.category
        )
        item_features_df = pd.merge(
            item_daily_features_df, item_categories_df, on="video_id"
        )
        del item_daily_features_df, item_categories_df

        return item_features_df

    def _create_item_daily_features_df(
        self, existing_video_ids: pd.Series, params: VideoDailyTableConfig
    ) -> pd.DataFrame:
        columns = self._get_features_columns(params=params.features)
        usecols = columns + ["video_id"]
        item_daily_features_df = pd.read_csv(params.data_path, usecols=usecols)
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

    def _create_item_categories_df(
        self, existing_video_ids: pd.Series, params: VideoCategoryTableConfig
    ) -> pd.DataFrame:
        columns = self._get_features_columns(params=params.features)
        usecols = columns + ["video_id"]
        item_categories_df = pd.read_csv(params.data_path, usecols=usecols)
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

    def _generate_logdata_using_observed_data(
        self,
        params: LogDataPropensityConfig,
        seed: int,
    ) -> pd.Series:
        usecols = ["user_id", "video_id"]
        observation_df = pd.read_csv(params.data_path, usecols=usecols)

        existing_video_ids = self.interaction_df["video_id"]
        isin_video_ids = observation_df["video_id"].isin(existing_video_ids)
        video_expo_counts = observation_df[isin_video_ids][
            "video_id"
        ].value_counts()

        existing_user_ids = self.interaction_df["user_id"]
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

        self.interaction_df["exposure"] = exposure_probabilitys
        # watch ratio >= 2を1とした基準で関連度を生成
        watch_ratio = self.interaction_df["watch_ratio"].values
        self.interaction_df["relevance"] = np.clip(watch_ratio / 2, 0, 1)
        self.interaction_df.drop("watch_ratio", axis=1, inplace=True)

        # 過去の推薦方策pi_bはランダムなポリシーとしてログデータを生成
        # ユーザの評価は時間に左右されないと仮定
        if params.behavior_policy == "random":
            np.random.seed(seed)
            self.interaction_df = self.interaction_df.sample(
                frac=1
            ).reset_index(drop=True)
            data_size = int(self.interaction_df.shape[0] * params.density)
            self.interaction_df = self.interaction_df.iloc[:data_size]
        else:
            raise ValueError("behavior_policy must be random")

        # generate clicks
        np.random.seed(seed)
        # バイアスのっかかったクリックデータを生成   P(Y = 1) = P(R = 1) * P(O = 1)
        self.interaction_df["biased_click"] = np.random.binomial(
            n=1,
            p=self.interaction_df["relevance"]
            * self.interaction_df["exposure"],
        )

        # テストデータ用のクリックデータ P(Y = 1) = P(R = 1)
        self.interaction_df["unbiased_click"] = np.random.binomial(
            n=1, p=self.interaction_df["relevance"]
        )

    def _get_features_columns(self, params: DictConfig) -> list:
        columns = OmegaConf.to_container(params, resolve=True)
        return list(columns.keys())
