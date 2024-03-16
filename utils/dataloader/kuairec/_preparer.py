# Standard library imports
from dataclasses import dataclass
from typing import Tuple, Dict

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Internal modules imports
from utils.dataloader.base import BaseLoader


@dataclass
class DatasetPreparer(BaseLoader):
    """prepare datasets for training and evaluation

    Args:
    - _seed (int): random seed
    """

    _seed: int

    def load(
        self,
        interaction_df: pd.DataFrame,
        features: csr_matrix,
    ) -> Tuple[dict, Dict[str, pd.DataFrame]]:
        """prepare datasets for training and evaluation

        Args:
        - interaction_df (pd.DataFrame): semi-synthetic log data for training and
        evaluation.
        - features (csr_matrix): features for training and evaluation.

        Returns:
            Tuple[dict, Dict[str, pd.DataFrame]]: datasets for training and evaluation
        """

        # split train, val, test
        dataframes, data_indices = self._split_datasets(interaction_df=interaction_df)

        # prepare fm_datasets
        fm_datasets = self._prepare_fm_datasets(
            features=features, feature_indices=data_indices
        )

        # prepare mf_datasets
        mf_datasets = self._prarpare_mf_datasets(dataframes=dataframes)
        # negative sampling
        train_sampled_indices = self._negative_sample(df=dataframes["train"])
        val_sampled_indices = self._negative_sample(df=dataframes["val"])

        datasets = {
            "FM": fm_datasets,
            "MF": mf_datasets,
            "sampled_train_indices": train_sampled_indices,
            "sampled_val_indices": val_sampled_indices,
        }

        return datasets, dataframes

    def _split_datasets(
        self, interaction_df: pd.DataFrame
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
        """split train, val, test datasets

        Args:
        - interaction_df (pd.DataFrame): semi-synthetic log data for training and
        evaluation.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]: train, val, test
        """

        datatypes = ["train", "val", "test"]
        data_indices = {}
        dataframes = {}
        for datatype in datatypes:
            data_filter = interaction_df["datatype"] == datatype
            _df = interaction_df[data_filter]
            data_indices[datatype] = _df.index.values
            dataframes[datatype] = _df.reset_index(drop=True).copy()

            if datatype == "test":
                dataframes[datatype]["label"] = dataframes[datatype]["relevance"]

        return dataframes, data_indices

    def _negative_sample(
        self, df: pd.DataFrame, negative_multiple: int = 1
    ) -> np.ndarray:
        """method to negative sample

        Args:
        - df (pd.DataFrame): train or val dataframe
        - negative_multiple (int, optional): multiple of negative sampling.
        Defaults to 1.

        Returns:
            np.ndarray: sampled indices
        """

        # negative sample
        positive_filter = df["label"] == 1
        positive_indices = df[positive_filter].index.values
        negative_indices = df[~positive_filter].index.values

        np.random.seed(self._seed)
        negative_indices = np.random.permutation(negative_indices)[
            : len(positive_indices) * int(negative_multiple)
        ]
        data_indices = np.r_[positive_indices, negative_indices]

        return data_indices

    def _prepare_fm_datasets(
        self,
        features: csr_matrix,
        feature_indices: Dict[str, np.ndarray],
    ) -> Dict[str, csr_matrix]:
        """prepare fm_datasets

        Args:
        - features (csr_matrix): features for training and evaluation.
        - feature_indices (Dict[str, np.ndarray]): feature indices for train, val, test.

        Returns:
            Dict[str, csr_matrix]: fm_datasets
        """

        datasets = {}
        for datatype, indices in feature_indices.items():
            datasets[datatype] = features[indices]

        return datasets

    def _prarpare_mf_datasets(
        self,
        dataframes: Dict[str, pd.DataFrame],
    ) -> dict:
        """prepare mf_datasets

        Args:
        - dataframes (Dict[str, pd.DataFrame]): train, val, test dataframes

        Returns:
            dict: mf_datasets
        """

        use_cols = ["user", "item"]
        datasets = {}
        for _df_name, _df in dataframes.items():
            datasets[_df_name] = _df[use_cols].values

        return datasets
