# Standard library imports
from dataclasses import dataclass
from typing import Dict, Tuple
from logging import Logger
from collections import defaultdict

# Third-party library imports
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

# Internal modules imports
from conf.kuairec import TableConfig
from utils.dataloader.kuairec._kuairec import KuaiRecCSVLoader
from utils.dataloader.base import BaseLoader

VALUE_ERROR_MESSAGE = "'{}' datatype is not supported. feature_name: '{}'"


@dataclass
class FeatureGenerator(BaseLoader):
    """generate features for training and evaluation

    Args:
    - _params (TableConfig): Configuration parameters for the KuaiRec dataset
    (read-only)
    - logger (Logger): Logger class instance
    """

    _params: TableConfig
    logger: Logger

    def load(
        self,
        interaction_df: pd.DataFrame,
    ) -> Tuple[csr_matrix, pd.DataFrame]:
        """generate features for training and evaluation

        Args:
        - interaction_df (pd.DataFrame): semi-synthetic log data for training and
        evaluation
        """

        dataframes_dict = self._create_basedict(
            interaction_df=interaction_df,
        )
        basefeatures, interaction_df = self._create_basefeature(
            interaction_df=interaction_df
        )
        tables_dict = OmegaConf.to_container(self._params)

        features = [basefeatures]
        for df_name, df in dataframes_dict.items():
            tables = tables_dict[df_name]

            if df_name == "video":
                columns = (
                    tables["daily"]["used_features"]
                    | tables["category"]["used_features"]
                )
            else:
                columns = tables["used_features"]

            converted_df = self._feature_engineering(df=df, columns=columns)
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

        return features, interaction_df

    def _feature_engineering(
        self, df: pd.DataFrame, columns: Dict[str, str]
    ) -> pd.DataFrame:
        """method to perform feature engineering

        Args:
        - df (pd.DataFrame): DataFrame to perform feature engineering
        - columns (Dict[str, str]): Dictionary of column names and their data types

        Raises:
            ValueError: If the datatype is not supported

        Returns:
            pd.DataFrame: DataFrame with feature engineering applied
        """

        datatypes = defaultdict(list)
        for feature_name, datatype in columns.items():
            if datatype not in {"int", "float", "label", "multilabel"}:
                error_message = VALUE_ERROR_MESSAGE.format(datatype, feature_name)
                self.logger.error(error_message)
                raise ValueError(error_message)

            datatypes[datatype].append(feature_name)

        if datatypes["label"]:
            df = pd.get_dummies(df, columns=datatypes["label"], dtype=int)

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

    def _create_basedict(
        self,
        interaction_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Create a dictionary of DataFrames for the interaction, user, and video

        Args:
        - interaction_df (pd.DataFrame): semi-synthetic log data for training and
        evaluation

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames for the interaction
        """

        user_features_df = KuaiRecCSVLoader.create_user_features_df(
            existing_user_ids=interaction_df["user_id"],
            _params=self._params.user,
            logger=self.logger,
        )
        item_features_df = KuaiRecCSVLoader.create_item_features_df(
            existing_video_ids=interaction_df["video_id"],
            _params=self._params.video,
            logger=self.logger,
        )

        dataframes_dict = {
            "interaction": interaction_df,
            "user": user_features_df,
            "video": item_features_df,
        }

        return dataframes_dict

    def _create_basefeature(
        self, interaction_df: pd.DataFrame
    ) -> Tuple[csr_matrix, pd.DataFrame]:
        """Create base features for training and evaluation

        Args:
        - interaction_df (pd.DataFrame): semi-synthetic log data for training and
        evaluation

        Returns:
            Tuple[csr_matrix, pd.DataFrame]: Tuple of base features and interaction
        """

        existing_unique_user_ids = interaction_df["user_id"].unique()
        existing_unique_video_ids = interaction_df["video_id"].unique()

        user_id2_index = {}
        for i, user_id in enumerate(np.sort(existing_unique_user_ids)):
            user_id2_index[user_id] = i

        interaction_df["user"] = interaction_df["user_id"].apply(
            lambda x: user_id2_index[x]
        )

        video_id2_index = {}
        for i, video_id in enumerate(np.sort(existing_unique_video_ids)):
            video_id2_index[video_id] = i

        interaction_df["item"] = interaction_df["video_id"].apply(
            lambda x: video_id2_index[x]
        )

        sparse_user_indices = csr_matrix(
            pd.get_dummies(interaction_df["user"], dtype=int).values
        )
        sparse_video_indices = csr_matrix(
            pd.get_dummies(interaction_df["item"], dtype=int).values
        )
        basefeatures = hstack([sparse_user_indices, sparse_video_indices])

        return basefeatures, interaction_df
