# Standard library imports
from dataclasses import dataclass
from typing import Tuple, Dict

# Third-party library imports
import numpy as np
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
import pandas as pd

# Internal modules imports
from conf.coat import LogDataConfig
from utils.dataloader.base import BaseLoader


@dataclass
class DatasetPreparer(BaseLoader):
    """Prepare the Coat dataset for FM and MF training

    Args:
    - seed (int): Random seed (read-only)
    - params (LogDataConfig): Configuration parameters for the Coat dataset (read-only)
    - pow_used (float): The power used to transform the estimated pscore (read-only)
    """

    _seed: int
    _params: LogDataConfig
    _pow_used: float

    def load(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> Tuple[dict, Dict[str, pd.DataFrame]]:
        """Prepare the Coat dataset for FM and MF training

        Args:
            train_df (pd.DataFrame): Biased Training data
            test_df (pd.DataFrame): Unbiased Test data
            user_features_df (pd.DataFrame): User features
            item_features_df (pd.DataFrame): Item features

        Returns:
            Tuple[dict, Dict[str, pd.DataFrame]]: Datasets and dataframes
        """

        # sparse user item id
        num_users = user_features_df.index.max() + 1
        num_items = item_features_df.index.max() + 1

        onehot_user_ids = sp.identity(num_users, format="csr")
        onehot_item_ids = sp.identity(num_items, format="csr")

        # sparse user item features
        user_features = sp.csr_matrix(user_features_df.values)
        item_features = sp.csr_matrix(item_features_df.values)

        # estimated pscore
        train_df["pscore"] = train_df["pscore"] ** self._pow_used

        # split train, val, test
        train_df, val_df = self._split_datasets(train_df)

        # negative sampling
        sampled_train_indices = self._nagative_sampling(train_df)
        sampled_val_indices = self._nagative_sampling(val_df)

        dfs = {"train": train_df, "val": val_df, "test": test_df}
        # prepare fm_datasets
        fm_features = self._get_fm_features(
            dfs=dfs,
            onehot_user_ids=onehot_user_ids,
            onehot_item_ids=onehot_item_ids,
            user_features=user_features,
            item_features=item_features,
        )

        # prepare mf_datasets
        mf_features = self._get_mf_features(dfs)

        datasets = {
            "FM": fm_features,
            "MF": mf_features,
            "item_embeddings": item_features,
            "sampled_train_indices": sampled_train_indices,
            "sampled_val_indices": sampled_val_indices,
        }

        return datasets, dfs

    def _split_datasets(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Split the train data into train and validation data

        Args:
            train_df (pd.DataFrame): Biased Training data

        Returns:
            Tuple[pd.DataFrame, ...]: Train and Validation data
        """

        train_df, val_df = train_test_split(
            train_df, test_size=self._params.val_ratio, random_state=self._seed
        )
        train_df: pd.DataFrame = train_df.reset_index(drop=True)
        val_df: pd.DataFrame = val_df.reset_index(drop=True)

        return train_df, val_df

    def _nagative_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Negative sampling

        Args:
            df (pd.DataFrame): train or val data

        Returns:
            pd.DataFrame: Negative sampled data
        """

        positive_filter = df["label"] == 1
        positive_indices = df[positive_filter].index.values
        negative_indices = df[~positive_filter].index.values

        np.random.seed(self._seed)
        negative_indices = np.random.permutation(negative_indices)[
            : len(positive_indices)
        ]
        data_indices = np.r_[positive_indices, negative_indices]

        return data_indices

    def _get_fm_features(
        self,
        dfs: Dict[str, pd.DataFrame],
        onehot_user_ids: sp.csr_matrix,
        onehot_item_ids: sp.csr_matrix,
        user_features: sp.csr_matrix,
        item_features: sp.csr_matrix,
    ) -> Dict[str, sp.csr_matrix]:
        """Prepare FM features

        Args:
            dfs (Dict[str, pd.DataFrame]): train, val, test data
            onehot_user_ids (sp.csr_matrix): onehot user ids
            onehot_item_ids (sp.csr_matrix): onehot item ids
            user_features (sp.csr_matrix): user features
            item_features (sp.csr_matrix): item features

        Returns:
            Dict[str, sp.csr_matrix]: FM features
        """

        fm_features = {}
        for _df_name, _df in dfs.items():
            csr_features = []

            # user
            user_ids = _df["user"].values
            csr_features.append(onehot_user_ids[user_ids])
            csr_features.append(user_features[user_ids])

            # item
            item_ids = _df["item"].values
            csr_features.append(onehot_item_ids[item_ids])
            csr_features.append(item_features[item_ids])

            fm_features[_df_name] = sp.hstack(csr_features)

        return fm_features

    def _get_mf_features(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Prepare MF features

        Args:
            dfs (Dict[str, pd.DataFrame]): train, val, test data

        Returns:
            Dict[str, np.ndarray]: MF features
        """

        mf_features = {}
        for _df_name, _df in dfs.items():
            mf_features[_df_name] = _df[["user", "item"]].values

        return mf_features
