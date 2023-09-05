import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from ast import literal_eval
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    StandardScaler,
)


def data_loader():
    interaction_df = _create_interaction_df()
    user_features_df = _create_user_features_df(
        user_ids=interaction_df["user_id"]
    )

    features = csr_matrix(
        pd.get_dummies(
            interaction_df["user_index"], drop_first=True, dtype=int
        ).values
    )

    video_indices = csr_matrix(
        pd.get_dummies(
            interaction_df["video_index"], drop_first=True, dtype=int
        ).values
    )
    features = hstack([features, video_indices])

    use_cols = [
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat6",
        "onehot_feat11",
        "onehot_feat12",
        "onehot_feat13",
        "onehot_feat14",
    ]
    user_features_df = pd.get_dummies(
        user_features_df, columns=use_cols, drop_first=True, dtype=int
    )
    sparse_user_features_df = pd.merge(
        interaction_df[["user_id"]], user_features_df, on="user_id", how="left"
    )
    sparse_user_features_df.drop(
        ["user_id", "register_days"], axis=1, inplace=True
    )
    features = hstack([features, csr_matrix(sparse_user_features_df.values)])
    del sparse_user_features_df

    usecols = [
        "video_id",
        "play_progress",
        "like_cnt",
        "share_user_num",
        "video_duration",
    ]
    item_daily_features_df = pd.read_csv(
        "../data/item_daily_features.csv", usecols=usecols
    )
    item_daily_features_df = item_daily_features_df.groupby("video_id").first()
    isin_video_ids = item_daily_features_df.index.isin(
        interaction_df["video_id"]
    )
    item_daily_features_df = item_daily_features_df[isin_video_ids]
    item_daily_features_df.rename_axis("index", inplace=True)
    item_daily_features_df["video_id"] = item_daily_features_df.index.values
    item_daily_features_df.reset_index(inplace=True)
    item_daily_features_df.drop("index", axis=1, inplace=True)

    item_categories_df = pd.read_csv("../data/item_categories.csv")
    isin_video_ids = item_categories_df["video_id"].isin(
        interaction_df["video_id"]
    )
    item_categories_df = item_categories_df[isin_video_ids]
    video_ids = item_categories_df["video_id"].values
    # 文字列から配列へ変換
    item_categories_df["feat"] = item_categories_df["feat"].apply(
        lambda x: literal_eval(x)
    )
    item_tags = item_categories_df["feat"].to_list()
    mlb = MultiLabelBinarizer()
    multi_hot_tags = mlb.fit_transform(item_tags)
    item_categories_df = pd.DataFrame(multi_hot_tags)
    item_categories_df["video_id"] = video_ids

    item_categories_df = pd.merge(
        interaction_df[["video_id"]],
        item_categories_df,
        on="video_id",
        how="left",
    )
    item_categories_df.drop(["video_id"], axis=1, inplace=True)
    features = hstack([features, csr_matrix(item_categories_df.values)])

    scaler = StandardScaler()
    # 欠損処理しないと使えない！！
    interaction_df[["timestamp"]] = scaler.fit_transform(
        interaction_df[["timestamp"]]
    )
    user_features_df[["register_days"]] = scaler.fit_transform(
        user_features_df[["register_days"]]
    )
    use_cols = [
        "video_duration",
        "play_progress",
        "like_cnt",
        "share_user_num",
    ]
    item_daily_features_df[use_cols] = scaler.fit_transform(
        item_daily_features_df[use_cols]
    )

    continuous_features_df = pd.merge(
        interaction_df[["user_id", "video_id", "timestamp"]],
        user_features_df[["user_id", "register_days"]],
        on="user_id",
        how="left",
    )
    continuous_features_df = pd.merge(
        continuous_features_df,
        item_daily_features_df,
        on="video_id",
        how="left",
    )

    continuous_features_df.drop(["user_id", "video_id"], axis=1, inplace=True)
    # ひとまず、列ごとの平均で埋める。(標準化したあとの)
    continuous_features_df.fillna(continuous_features_df.mean(), inplace=True)
    continuous_features = csr_matrix(continuous_features_df.values)
    features = hstack([features, continuous_features])

    # split_ratio = (0.8, 0.1, 0.1)
    # split_index = [int(data_size*r) for r in split_ratio[:2]]
    # split_index[1] += split_index[0]

    return features, interaction_df


def _create_interaction_df() -> pd.DataFrame:
    use_cols = ["user_id", "video_id", "timestamp", "watch_ratio"]
    interaction_df = pd.read_csv("../data/small_matrix.csv", usecols=use_cols)

    # 自然に観測されたbig_matrix上でのアイテム,ユーザ-の相対的な露出傾向を使い人工的なクリックデータを生成
    use_cols = ["user_id", "video_id"]
    obs_df = pd.read_csv("../data/big_matrix.csv", usecols=use_cols)

    isin_video_id = obs_df["video_id"].isin(interaction_df["video_id"])
    video_expo_counts = obs_df[isin_video_id]["video_id"].value_counts()
    isin_user_id = obs_df["user_id"].isin(interaction_df["user_id"])
    user_expo_counts = obs_df[isin_user_id]["user_id"].value_counts()
    del obs_df

    video_expo_probs = (video_expo_counts / video_expo_counts.max()) ** 0.5
    user_expo_probs = (user_expo_counts / user_expo_counts.max()) ** 0.5
    interaction_df["exposure"] = video_expo_probs[
        interaction_df["video_id"]
    ].values
    interaction_df["exposure"] *= user_expo_probs[
        interaction_df["user_id"]
    ].values

    # watch ratio >= 2を1とした基準で関連度を生成
    watch_ratio = interaction_df["watch_ratio"].values
    interaction_df["relevance"] = np.clip(watch_ratio / 2, 0, 1)

    np.random.seed(12345)
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

    # 過去の推薦方策pi_bはランダムなポリシーとしてログデータを生成
    # ユーザの評価は時間に左右されないと仮定
    interaction_df.drop("watch_ratio", axis=1, inplace=True)
    interaction_df = interaction_df.sample(frac=1).reset_index(drop=True)
    data_size = interaction_df.shape[0] // 20
    interaction_df = interaction_df.iloc[:data_size]

    return interaction_df


def _create_user_features_df(user_ids: pd.Series) -> pd.DataFrame:
    use_cols = [
        "user_id",
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat6",
        "onehot_feat11",
        "register_days",
        "onehot_feat12",
        "onehot_feat13",
        "onehot_feat14",
    ]
    user_features_df = pd.read_csv(
        "../data/user_features.csv", usecols=use_cols
    )
    isin_user_ids = user_features_df["user_id"].isin(user_ids)
    user_features_df = user_features_df[isin_user_ids]

    return user_features_df
