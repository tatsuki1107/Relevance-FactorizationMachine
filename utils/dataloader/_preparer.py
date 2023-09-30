from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple, Dict
from utils.model import FMDataset, MFDataset
from utils.dataloader.base import BaseLoader


@dataclass
class DatasetPreparer(BaseLoader):
    seed: int

    def load(
        self,
        interaction_df: pd.DataFrame,
        features: csr_matrix,
    ) -> dict:
        usecols = ["datatype", "biased_click"]
        # split train, val, test
        dataset_indices = self._split_datasets(df=interaction_df[usecols])

        # prepare fm_datasets
        fm_datasets = self._prepare_fm_datasets(
            features=features, feature_indices=dataset_indices
        )

        # prepare pmf_datasets
        usecols = ["user_index", "video_index"]
        mf_datasets = self._prarpare_mf_datasets(
            df=interaction_df[usecols], df_indices=dataset_indices
        )

        # prepare pscores and clicks
        usecols = ["exposure", "biased_click", "relevance", "unbiased_click"]
        relevances, pscores, clicks = self._prepare_pscores_and_clicks_rel(
            df=interaction_df[usecols], df_indices=dataset_indices
        )

        # prepare test_user2indices
        usecols = ["user_index", "exposure"]
        val_user2data_indices = self._create_user2data_indices(
            interaction_df=interaction_df[usecols],
            df_indices=dataset_indices["val"],
            frequency={"all"},
        )
        test_user2data_indices = self._create_user2data_indices(
            interaction_df=interaction_df[usecols],
            df_indices=dataset_indices["test"],
            frequency={"all", "popular", "rare"},
        )
        user2data_indices = {
            "val": val_user2data_indices,
            "test": test_user2data_indices,
        }

        datasets = {
            "FM": fm_datasets,
            "MF": mf_datasets,
            "relevances": relevances,
            "clicks": clicks,
            "pscores": pscores,
            "user2data_indices": user2data_indices,
        }

        return datasets

    def _split_datasets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        datatypes = ["train", "val", "test"]
        res = {}
        for datatype in datatypes:
            data_filter = df["datatype"] == datatype
            if datatype in {"train", "val"}:
                data_indices = self._negative_sample(df=df[data_filter])
            else:
                data_indices = df[data_filter].index.values
            res[datatype] = data_indices

        return res

    def _negative_sample(
        self, df: pd.DataFrame, negative_multiple: int = 2
    ) -> np.ndarray:
        # negative sample
        positive_filter = df["biased_click"] == 1
        positive_indices = df[positive_filter].index.values
        negative_indices = df[~positive_filter].index.values

        np.random.seed(self.seed)
        negative_indices = np.random.permutation(negative_indices)[
            : len(positive_indices) * negative_multiple
        ]
        data_indices = np.r_[positive_indices, negative_indices]

        return data_indices

    def _prepare_fm_datasets(
        self,
        features: csr_matrix,
        feature_indices: Dict[str, np.ndarray],
    ) -> FMDataset:
        datasets = {}
        for datatype, indices in feature_indices.items():
            datasets[datatype] = features[indices]

        return FMDataset(**datasets)

    def _prarpare_mf_datasets(
        self,
        df: pd.DataFrame,
        df_indices: Dict[str, np.ndarray],
    ) -> MFDataset:
        datasets = {}
        for datatype, indices in df_indices.items():
            datasets[datatype] = df.iloc[indices].values

        n_users = df["user_index"].max() + 1
        n_items = df["video_index"].max() + 1
        datasets["n_users"] = n_users
        datasets["n_items"] = n_items

        return MFDataset(**datasets)

    def _prepare_pscores_and_clicks_rel(
        self,
        df: pd.DataFrame,
        df_indices: Dict[str, np.ndarray],
    ) -> Tuple[dict]:
        relevances, pscores, clicks = {}, {}, {}
        for datatype, indices in df_indices.items():
            data_df = df.iloc[indices]

            if datatype in {"train", "val"}:
                pscores[datatype] = data_df["exposure"].values
                clicks[datatype] = data_df["biased_click"].values
                relevances[datatype] = data_df["relevance"].values
            else:
                clicks[datatype] = data_df["unbiased_click"].values

        return relevances, pscores, clicks

    def _create_user2data_indices(
        self,
        interaction_df: pd.DataFrame,
        df_indices: np.ndarray,
        frequency: set,
        thetahold: int = 0.75,
    ) -> Dict[str, list]:
        data_df = interaction_df.iloc[df_indices].reset_index(drop=True)
        dataframe_dict = {}
        for freq in frequency:
            if freq == "all":
                dataframe_dict[freq] = data_df

            elif freq == "rare":
                freq_filter = data_df["exposure"] <= thetahold
                dataframe_dict[freq] = data_df[freq_filter]

            elif freq == "popular":
                freq_filter = data_df["exposure"] > thetahold
                dataframe_dict[freq] = data_df[freq_filter]

        user2data_indices = self._get_data_indices(
            dataframe_dict=dataframe_dict
        )

        return user2data_indices

    def _get_data_indices(
        self, dataframe_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, list]:
        user2data_indices = {}
        for frequency, df in dataframe_dict.items():
            groups = df.groupby(["user_index"])
            df_indices_per_user = []
            for _, group in groups:
                df_indices_per_user.append(group.index.tolist())

            user2data_indices[frequency] = df_indices_per_user

        return user2data_indices
