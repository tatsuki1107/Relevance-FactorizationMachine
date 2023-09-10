import pandas as pd
import numpy as np
from utils.model import Dataset, Pscores
from omegaconf import OmegaConf
from conf.setting.default import (
    ExperimentConfig,
    UserTableConfig,
    VideoDailyTableConfig,
    LogDataPropensityConfig,
    VideoCategoryTableConfig,
    VideoTableConfig,
)
from collections import defaultdict
from typing import Tuple
from scipy.sparse import csr_matrix, hstack
from ast import literal_eval
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    StandardScaler,
)


def dataloader(params: ExperimentConfig):
    interaction_df, basefeatures = _create_interaction_df(params=params)
    user_features_df = _create_user_features_df(
        existing_user_ids=interaction_df["user_id"],
        params=params.tables.user,
    )
    item_features_df = _create_item_features_df(
        existing_video_ids=interaction_df["video_id"],
        params=params.tables.video,
    )

    dfs = {
        "interaction": interaction_df,
        "user": user_features_df,
        "video": item_features_df,
    }
    tables_dict = OmegaConf.to_container(params.tables, resolve=True)
    features = [basefeatures]
    for df_name, df in dfs.items():
        tables = tables_dict[df_name]

        if df_name == "video":
            columns = (
                tables["daily"]["features"] | tables["category"]["features"]
            )
        else:
            columns = tables["features"]

        converted_df = _feature_engineering(df=df, columns=columns)
        if df_name == "interaction":
            columns = list(columns.keys())
            sparse_features = csr_matrix(converted_df[columns].values)

        else:
            id = f"{df_name}_id"
            features_df = pd.merge(
                interaction_df[[id]],
                converted_df,
                on=id,
                how="left",
            )
            features_df.drop([id], axis=1, inplace=True)
            sparse_features = csr_matrix(features_df.values)

        features.append(sparse_features)

    features = hstack(features)
    interaction_df.drop(["user_id", "video_id"], axis=1, inplace=True)

    # split train, val, test
    data_size = interaction_df.shape[0]
    split_index = [
        int(data_size * ratio)
        for ratio in params.logdata_propensity.train_val_test_ratio[:2]
    ]
    split_index[1] += split_index[0]
    train_indices = np.arange(split_index[0])
    val_indices = np.arange(split_index[0], split_index[1])
    test_indices = np.arange(split_index[1], data_size)

    test_df = interaction_df.iloc[test_indices].reset_index(drop=True)
    groups = test_df.sort_values(by=["user_index", "unbiased_click"]).groupby(
        "user_index"
    )
    test_user2indices = []
    for _, group in groups:
        test_user2indices.append(group.index.tolist())

    # prepare data for FM and PMF
    indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    datasets = {}
    for _data, _indices in indices.items():
        datasets[_data] = features[_indices]

    fm_datasets = Dataset(**datasets)

    datasets = {}
    for _data, _indices in indices.items():
        columns = ["user_index", "video_index"]
        datasets[_data] = interaction_df.iloc[_indices][columns].values

    mf_datasets = Dataset(**datasets)

    pscores, clicks = {}, defaultdict(dict)
    for _data, _indices in indices.items():
        if _data in {"train", "val"}:
            pscores[_data] = interaction_df.iloc[_indices]["exposure"].values
            clicks[_data]["biased"] = interaction_df.iloc[_indices][
                "biased_click"
            ].values

        clicks[_data]["unbiased"] = interaction_df.iloc[_indices][
            "unbiased_click"
        ].values

    n_users = interaction_df["user_index"].max() + 1
    n_items = interaction_df["video_index"].max() + 1

    pscores = Pscores(**pscores)
    datasets = {
        "FM": fm_datasets,
        "PMF": mf_datasets,
        "n_users": n_users,
        "n_items": n_items,
    }

    return datasets, pscores, clicks, test_user2indices


def _create_interaction_df(
    params: ExperimentConfig,
) -> Tuple[pd.DataFrame, csr_matrix]:
    interaction = params.tables.interaction
    columns = _get_features_columns(params=interaction.features)
    usecols = columns + ["user_id", "video_id", "watch_ratio"]
    interaction_df = pd.read_csv(interaction.data_path, usecols=usecols)

    # 自然に観測されたbig_matrix上でのアイテム,ユーザ-の相対的な露出傾向を使い人工的なクリックデータを生成
    exposure_probabilitys = _generate_exposure_probability_using_big_matrix(
        existing_video_ids=interaction_df["video_id"],
        existing_user_ids=interaction_df["user_id"],
        params=params.logdata_propensity,
    )
    interaction_df["exposure"] = exposure_probabilitys

    # watch ratio >= 2を1とした基準で関連度を生成
    watch_ratio = interaction_df["watch_ratio"].values
    interaction_df["relevance"] = np.clip(watch_ratio / 2, 0, 1)
    interaction_df.drop("watch_ratio", axis=1, inplace=True)

    # 過去の推薦方策pi_bはランダムなポリシーとしてログデータを生成
    # ユーザの評価は時間に左右されないと仮定
    if params.logdata_propensity.behavior_policy == "random":
        np.random.seed(params.seed)
        interaction_df = interaction_df.sample(frac=1).reset_index(drop=True)
        data_size = int(
            interaction_df.shape[0] * params.logdata_propensity.density
        )
        interaction_df = interaction_df.iloc[:data_size]
    else:
        raise ValueError("behavior_policy must be random")

    # generate clicks
    np.random.seed(params.seed)
    # バイアスのっかかったクリックデータを生成   P(Y = 1) = P(R = 1) * P(O = 1)
    interaction_df["biased_click"] = np.random.binomial(
        n=1, p=interaction_df["relevance"] * interaction_df["exposure"]
    )

    # テストデータ用のクリックデータ P(Y = 1) = P(R = 1)
    interaction_df["unbiased_click"] = np.random.binomial(
        n=1, p=interaction_df["relevance"]
    )

    existing_unique_user_ids = interaction_df["user_id"].unique()
    existing_unique_video_ids = interaction_df["video_id"].unique()

    user_id2_index = {}
    for i, user_id in enumerate(np.sort(existing_unique_user_ids)):
        user_id2_index[user_id] = i

    # 観測されたユーザIDを0から順に振り直す
    interaction_df["user_index"] = interaction_df["user_id"].apply(
        lambda x: user_id2_index[x]
    )

    video_id2_index = {}
    for i, video_id in enumerate(np.sort(existing_unique_video_ids)):
        video_id2_index[video_id] = i

    # 観測された動画IDを0から順に振り直す
    interaction_df["video_index"] = interaction_df["video_id"].apply(
        lambda x: video_id2_index[x]
    )

    sparse_user_indices = csr_matrix(
        pd.get_dummies(
            interaction_df["user_index"], drop_first=True, dtype=int
        ).values
    )
    sparse_video_indices = csr_matrix(
        pd.get_dummies(
            interaction_df["video_index"], drop_first=True, dtype=int
        ).values
    )
    basefeatures = hstack([sparse_user_indices, sparse_video_indices])

    return interaction_df, basefeatures


def _get_features_columns(params) -> list:
    columns = OmegaConf.to_container(params, resolve=True)
    return list(columns.keys())


def _create_user_features_df(
    existing_user_ids: pd.Series, params: UserTableConfig
) -> pd.DataFrame:
    columns = _get_features_columns(params=params.features)
    usecols = columns + ["user_id"]
    user_features_df = pd.read_csv(params.data_path, usecols=usecols)
    isin_user_ids = user_features_df["user_id"].isin(existing_user_ids)
    user_features_df = user_features_df[isin_user_ids].reset_index(drop=True)

    return user_features_df


def _create_item_features_df(
    existing_video_ids: pd.Series, params: VideoTableConfig
) -> pd.DataFrame:
    item_daily_features_df = _create_item_daily_features_df(
        existing_video_ids=existing_video_ids, params=params.daily
    )

    item_categories_df = _create_item_categories_df(
        existing_video_ids=existing_video_ids, params=params.category
    )
    item_features_df = pd.merge(
        item_daily_features_df, item_categories_df, on="video_id"
    )
    del item_daily_features_df, item_categories_df

    return item_features_df


def _feature_engineering(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    datatypes = defaultdict(list)
    for feature_name, datatype in columns.items():
        datatypes[datatype].append(feature_name)

    if datatypes["label"]:
        df = pd.get_dummies(
            df, columns=datatypes["label"], drop_first=True, dtype=int
        )

    if datatypes["int"] or datatypes["float"]:
        usecols = datatypes["int"] + datatypes["float"]
        scaler = StandardScaler()
        df[usecols] = scaler.fit_transform(df[usecols])
        # 欠損値はひとまず平均値で埋める
        df[usecols] = df[usecols].fillna(df[usecols].mean())

    if datatypes["multilabel"]:
        for col in datatypes["multilabel"]:
            multilabels = df[col].to_list()
            mlb = MultiLabelBinarizer()
            multi_hot_tags = mlb.fit_transform(multilabels)
            item_tags_df = pd.DataFrame(multi_hot_tags)
            df = pd.concat([df, item_tags_df], axis=1)
            df.drop(col, axis=1, inplace=True)

    return df


def _create_item_daily_features_df(
    existing_video_ids: pd.Series, params: VideoDailyTableConfig
) -> pd.DataFrame:
    columns = _get_features_columns(params=params.features)
    usecols = columns + ["video_id"]
    item_daily_features_df = pd.read_csv(params.data_path, usecols=usecols)
    item_daily_features_df = item_daily_features_df.groupby("video_id").first()
    isin_video_ids = item_daily_features_df.index.isin(existing_video_ids)
    item_daily_features_df = item_daily_features_df[isin_video_ids]
    item_daily_features_df.rename_axis("index", inplace=True)
    item_daily_features_df["video_id"] = item_daily_features_df.index.values
    item_daily_features_df.reset_index(inplace=True, drop=True)

    return item_daily_features_df


def _create_item_categories_df(
    existing_video_ids: pd.Series, params: VideoCategoryTableConfig
) -> pd.DataFrame:
    columns = _get_features_columns(params=params.features)
    usecols = columns + ["video_id"]
    item_categories_df = pd.read_csv(params.data_path, usecols=usecols)
    isin_video_ids = item_categories_df["video_id"].isin(existing_video_ids)
    item_categories_df = item_categories_df[isin_video_ids].reset_index(
        drop=True
    )
    # 文字列から配列へ変換
    item_categories_df["feat"] = item_categories_df["feat"].apply(
        lambda x: literal_eval(x)
    )
    return item_categories_df


def _generate_exposure_probability_using_big_matrix(
    existing_video_ids: pd.Series,
    existing_user_ids: pd.Series,
    params: LogDataPropensityConfig,
) -> pd.Series:
    usecols = ["user_id", "video_id"]
    obs_df = pd.read_csv("./data/kuairec/big_matrix.csv", usecols=usecols)

    isin_video_ids = obs_df["video_id"].isin(existing_video_ids)
    video_expo_counts = obs_df[isin_video_ids]["video_id"].value_counts()

    isin_user_ids = obs_df["user_id"].isin(existing_user_ids)
    user_expo_counts = obs_df[isin_user_ids]["user_id"].value_counts()
    del obs_df

    video_expo_probs = (
        video_expo_counts / video_expo_counts.max()
    ) ** params.exposure_bias
    user_expo_probs = (
        user_expo_counts / user_expo_counts.max()
    ) ** params.exposure_bias
    exposure_probabilitys = video_expo_probs[existing_video_ids].values
    exposure_probabilitys *= user_expo_probs[existing_user_ids].values

    return exposure_probabilitys
