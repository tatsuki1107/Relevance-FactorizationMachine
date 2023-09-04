import pandas as pd


def data_loader():
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
