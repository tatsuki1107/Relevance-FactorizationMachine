import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, hstack
from ast import literal_eval
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    StandardScaler,
)


def dataloader(params):
    interaction_df, basefeatures = _create_interaction_df()
    user_features_df = _create_user_features_df(
        existing_user_ids=interaction_df["user_id"], params=params
    )
    item_features_df = _create_item_features_df(
        existing_video_ids=interaction_df["video_id"], params=params
    )

    dfs = {
        "interaction": interaction_df,
        "user": user_features_df,
        "video": item_features_df,
    }
    features = [basefeatures]
    for df_name, df in dfs.items():
        columns = params.features.__dict__[df_name]
        if df_name == "video":
            columns = columns["daily"] | columns["category"]

        converted_df = _feature_engineering(df=df, columns=columns)
        if df_name == "interaction":
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

    return interaction_df, features


def _create_interaction_df(params) -> pd.DataFrame:
    columns = params.features.interaction.__dict__
    usecols = list(columns.keys()) + ["user_id", "video_id", "watch_ratio"]
    interaction_df = pd.read_csv("../data/small_matrix.csv", usecols=usecols)

    # 自然に観測されたbig_matrix上でのアイテム,ユーザ-の相対的な露出傾向を使い人工的なクリックデータを生成
    exposure_probabilitys = _generate_exposure_probability_using_big_matrix(
        existing_video_ids=interaction_df["video_id"],
        existing_user_ids=interaction_df["user_id"],
        params=params,
    )
    interaction_df["exposure"] = exposure_probabilitys

    # watch ratio >= 2を1とした基準で関連度を生成
    watch_ratio = interaction_df["watch_ratio"].values
    interaction_df["relevance"] = np.clip(watch_ratio / 2, 0, 1)
    interaction_df.drop("watch_ratio", axis=1, inplace=True)

    # 過去の推薦方策pi_bはランダムなポリシーとしてログデータを生成
    # ユーザの評価は時間に左右されないと仮定
    if params.behavior_policy == "random":
        np.random.seed(params.seed)
        interaction_df = interaction_df.sample(frac=1).reset_index(drop=True)
        data_size = interaction_df.shape[0] // 20
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


def _create_user_features_df(
    existing_user_ids: pd.Series, params
) -> pd.DataFrame:
    columns = params.user.__dict__
    usecols = list(columns.keys()) + ["user_id"]
    user_features_df = pd.read_csv(
        "../data/user_features.csv", usecols=usecols
    )
    isin_user_ids = user_features_df["user_id"].isin(existing_user_ids)
    user_features_df = user_features_df[isin_user_ids].reset_index(drop=True)

    return user_features_df


def _create_item_features_df(
    existing_video_ids: pd.Series, params
) -> pd.DataFrame:
    item_daily_features_df = _create_item_daily_features_df(
        existing_video_ids=existing_video_ids, params=params
    )

    item_categories_df = _create_item_categories_df(
        existing_video_ids=existing_video_ids, params=params
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
        df[usecols].fillna(df[usecols].mean(), inplace=True)

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
    existing_video_ids: pd.Series, params
) -> pd.DataFrame:
    columns = params.daily.__dict__
    usecols = list(columns.keys()) + ["video_id"]
    item_daily_features_df = pd.read_csv(
        "../data/item_daily_features.csv", usecols=usecols
    )
    item_daily_features_df = item_daily_features_df.groupby("video_id").first()
    isin_video_ids = item_daily_features_df.index.isin(existing_video_ids)
    item_daily_features_df = item_daily_features_df[isin_video_ids]
    item_daily_features_df.rename_axis("index", inplace=True)
    item_daily_features_df["video_id"] = item_daily_features_df.index.values
    item_daily_features_df.reset_index(inplace=True, drop=True)

    return item_daily_features_df


def _create_item_categories_df(
    existing_video_ids: pd.Series, params
) -> pd.DataFrame:
    columns = params.category.__dict__
    usecols = list(columns.keys()) + ["video_id"]
    item_categories_df = pd.read_csv(
        "../data/item_categories.csv", usecols=usecols
    )
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
    existing_video_ids: pd.Series, existing_user_ids: pd.Series, params
) -> pd.Series:
    use_cols = ["user_id", "video_id"]
    obs_df = pd.read_csv("../data/big_matrix.csv", usecols=use_cols)

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
