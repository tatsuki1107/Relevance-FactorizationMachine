# Standard library imports
from dataclasses import dataclass
from typing import Tuple, Dict, Union
from logging import Logger

# Third-party library imports
import numpy as np
from scipy.sparse import csr_matrix

# Internal modules imports
from conf.coat import ExperimentConfig
from utils.dataloader.base import BaseLoader
from utils.dataloader.coat._preparer import DatasetPreparer
from utils.dataloader.coat._coat import CoatCSVLoader

MODEL_NAME_ERROR_MESSAGE = "model_name must be FM or MF. model_name: '{}'"
ESTIMATOR_NAME_ERROR_MESSAGE = "estimator must be IPS or Naive. estimator: '{}'"


@dataclass
class DataLoader(BaseLoader):
    """load data for training and evaluation

    Args:
    - params (ExperimentConfig): Configuration parameters for the Coat dataset
    (read-only)
    - logger (Logger): Logger class instance
    """

    _params: ExperimentConfig
    logger: Logger

    def __post_init__(self) -> None:
        csv_loader = CoatCSVLoader(
            _params=self._params.tables, _seed=self._params.seed, logger=self.logger
        )

        train_df, test_df, user_features_df, item_features_df = csv_loader.load()
        self.logger.info("loaded csv files")

        preparer = DatasetPreparer(
            _seed=self._params.seed,
            _params=self._params.data_logging_settings,
            _pow_used=self._params.pow_used,
        )
        self.datasets, self.dataframes = preparer.load(
            train_df=train_df,
            test_df=test_df,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
        )
        self.n_users: int = user_features_df.index.max() + 1
        self.n_items: int = item_features_df.index.max() + 1

        self.logger.info(
            f'train_size: {len(self.dataframes["train"])}, '
            f"sampled_train_size: {len(self.datasets['sampled_train_indices'])}, "
            f"val_size: {len(self.dataframes['val'])}, "
            f"sampled_val_size: {len(self.datasets['sampled_val_indices'])}, "
            f' test_size: {len(self.dataframes["test"])}, '
        )
        self.logger.info("prepared train and evaluation datasets")

    def load(
        self, model_name: str, estimator: str
    ) -> Tuple[Dict[str, Union[csr_matrix, np.ndarray]]]:
        """load data for training and evaluation

        Args:
            model_name (str): model name (FM or MF)
            estimator (str): estimator name (IPS or Naive)

        Returns:
            Tuple[Dict[str, Union[csr_matrix, np.ndarray]]]: train and val data
        """

        _validate_params(model_name=model_name, estimator=estimator)

        # labels
        train_y, val_y = self._get_labels()

        # pscores
        (
            train_pscores,
            val_pscores,
        ) = self._get_pscores(estimator=estimator)

        # features
        features = self.datasets[model_name]

        sampled_indices = self.datasets["sampled_train_indices"]
        train = {
            "features": features["train"][sampled_indices],
            "labels": train_y[sampled_indices],
            "pscores": train_pscores[sampled_indices],
        }
        sampled_indices = self.datasets["sampled_val_indices"]
        val = {
            "features": features["val"][sampled_indices],
            "labels": val_y[sampled_indices],
            "pscores": val_pscores[sampled_indices],
        }

        return train, val

    def _get_labels(self) -> Tuple[np.ndarray, ...]:
        """Get labels for training and evaluation

        Returns:
            Tuple[np.ndarray, ...]: labels for training and evaluation
        """

        labels = []
        for _df_name, _df in self.dataframes.items():
            if _df_name in {"train", "val"}:
                labels.append(_df["label"].values)

        return labels

    def _get_pscores(self, estimator: str) -> Tuple[np.ndarray, ...]:
        """Get pscores for training and evaluation

        Args:
            estimator (str): estimator name (IPS or Naive)

        Returns:
            Tuple[np.ndarray, ...]: pscores for training and evaluation
        """

        col_name = "pscore" if estimator == "IPS" else "ones_pscore"
        pscores = []
        for _df_name, _df in self.dataframes.items():
            if _df_name in {"train", "val"}:
                pscores.append(_df[col_name].values)

        return pscores

    @property
    def val_df(self):
        """Validation data"""
        return self.dataframes["val"]

    @property
    def val_evaluation_features(self):
        """Validation data for evaluation"""
        return dict(MF=self.datasets["MF"]["val"], FM=self.datasets["FM"]["val"])

    @property
    def test_df(self):
        """Test data"""
        return self.dataframes["test"]

    @property
    def test_evaluation_features(self):
        """Test data for evaluation"""
        return dict(MF=self.datasets["MF"]["test"], FM=self.datasets["FM"]["test"])

    @property
    def item_populality(self):
        """Item popularity"""
        item_populalities = (
            self.dataframes["train"]
            .groupby("item")
            .agg({"label": "count"})
            .values.reshape(-1)
        )
        item_populalities = item_populalities / item_populalities.max()

        return item_populalities


def _validate_params(model_name: str, estimator: str) -> None:
    if model_name not in ["FM", "MF"]:
        raise ValueError(MODEL_NAME_ERROR_MESSAGE.format(model_name))

    if estimator not in ["IPS", "Naive"]:
        raise ValueError(ESTIMATOR_NAME_ERROR_MESSAGE.format(estimator))
