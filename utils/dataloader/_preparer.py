from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple
from collections import defaultdict
from utils.model import FMDataset, PMFDataset


@dataclass
class DatasetPreparer:
    def prepare_dataset(
        self,
        interaction_df: pd.DataFrame,
        features: csr_matrix,
        train_val_test_ratio: list,
    ) -> dict:
        # split train, val, test
        data_size = interaction_df.shape[0]
        split_index = [
            int(data_size * ratio) for ratio in train_val_test_ratio[:2]
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
            features=features, dataset_indices=dataset_indices
        )

        # prepare pmf_datasets
        pmf_datasets = self._prarpare_pmf_datasets(
            interaction_df=interaction_df, dataset_indices=dataset_indices
        )

        # prepare pscores and clicks
        pscores, clicks = self._prepare_pscores_and_clicks(
            interaction_df=interaction_df,
            dataset_indices=dataset_indices,
        )

        # prepare test_user2indices
        test_user2indices = self._create_test_user2indices(
            interaction_df=interaction_df,
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
        features: csr_matrix,
        dataset_indices: dict,
    ) -> FMDataset:
        datasets = {}
        for _datatype, _indices in dataset_indices.items():
            datasets[_datatype] = features[_indices]

        fm_datasets = FMDataset(**datasets)

        return fm_datasets

    def _prarpare_pmf_datasets(
        self,
        interaction_df: pd.DataFrame,
        dataset_indices: dict,
    ) -> PMFDataset:
        datasets = {}
        columns = ["user_index", "video_index"]
        for _datatype, _indices in dataset_indices.items():
            datasets[_datatype] = interaction_df.iloc[_indices][columns].values

        n_users = interaction_df["user_index"].max() + 1
        n_items = interaction_df["video_index"].max() + 1
        datasets["n_users"] = n_users
        datasets["n_items"] = n_items

        pmf_datasets = PMFDataset(**datasets)

        return pmf_datasets

    def _prepare_pscores_and_clicks(
        self,
        interaction_df: pd.DataFrame,
        dataset_indices: dict,
    ) -> Tuple[dict, defaultdict]:
        pscores, clicks = {}, defaultdict(dict)
        for _data, _indices in dataset_indices.items():
            if _data in {"train", "val"}:
                pscores[_data] = interaction_df.iloc[_indices][
                    "exposure"
                ].values
                clicks[_data]["biased"] = interaction_df.iloc[_indices][
                    "biased_click"
                ].values
                clicks[_data]["unbiased"] = interaction_df.iloc[_indices][
                    "relevance"
                ].values
            else:
                clicks[_data]["unbiased"] = interaction_df.iloc[_indices][
                    "unbiased_click"
                ].values

        return pscores, clicks

    def _create_test_user2indices(
        self,
        interaction_df: pd.DataFrame,
        test_indices: np.ndarray,
        thetahold: int = 0.3,
    ) -> dict:
        # all
        test_df = interaction_df.iloc[test_indices].reset_index(drop=True)

        filter = test_df["exposure"] <= thetahold
        # rare (low exposure)
        rare_test_df = test_df[filter]

        # popular (high exposure)
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
