# Standard library imports
from typing import Tuple
from logging import Logger
from dataclasses import dataclass

# Third-party library imports
import numpy as np

# Internal modules imports
from conf.kuairec import ExperimentConfig
from utils.dataloader.base import BaseLoader
from utils.dataloader.kuairec._kuairec import KuaiRecCSVLoader
from utils.dataloader.kuairec._preparer import DatasetPreparer
from utils.dataloader.kuairec._click import SemiSyntheticLogDataGenerator
from utils.dataloader.kuairec._feature import FeatureGenerator


MODEL_NAME_ERROR_MESSAGE = "model_name must be FM or MF. model_name: '{}'"
ESTIMATOR_NAME_ERROR_MESSAGE = "estimator must be Ideal, IPS or Naive. estimator: '{}'"


@dataclass
class DataLoader(BaseLoader):
    """load data for training and evaluation

    Args:
    - _params (ExperimentConfig): Configuration parameters for the KuaiRec dataset
    (read-only)
    - logger (Logger): Logger class instance
    """

    _params: ExperimentConfig
    logger: Logger

    def __post_init__(self) -> None:
        """load semi-synthetic log data for training and evaluation"""

        small_matrix_df = KuaiRecCSVLoader.create_small_matrix_df(
            _params=self._params.tables.interaction,
            logger=self.logger,
        )
        self.logger.info("created small_matrix_df")

        logdata_generator = SemiSyntheticLogDataGenerator(
            _seed=self._params.seed,
            _params=self._params.data_logging_settings,
            logger=self.logger,
        )
        interaction_df = logdata_generator.load(
            interaction_df=small_matrix_df,
        )
        self.logger.info("created interaction_df")

        feature_generator = FeatureGenerator(
            _params=self._params.tables,
            logger=self.logger,
        )
        features, interaction_df = feature_generator.load(interaction_df=interaction_df)
        self.n_users = interaction_df["user"].max() + 1
        self.n_items = interaction_df["item"].max() + 1
        self.logger.info("created features")

        preparer = DatasetPreparer(_seed=self._params.seed)
        self.datasets, self.dataframes = preparer.load(
            interaction_df=interaction_df,
            features=features,
        )

        self.logger.info(
            f'train_size: {len(self.dataframes["train"])}, '
            f"sampled_train_size: {len(self.datasets['sampled_train_indices'])}, "
            f"val_size: {len(self.dataframes['val'])}, "
            f"sampled_val_size: {len(self.datasets['sampled_val_indices'])}, "
            f' test_size: {len(self.dataframes["test"])}, '
        )
        self.logger.info("prepared train and evaluation datasets")

    def load(self, model_name: str, estimator: str) -> Tuple[list]:
        """Return training and validation data for each model and estimator

        Args:
        - model_name (str): Model name (FM or MF)
        - estimator (str): Estimator name (IPS or Naive)

        Raises:
        - ValueError: model_name must be FM or MF
        - ValueError: estimator must be IPS or Naive

        Returns:
        - (tuple): train and val data
        """

        self._validate_params(model_name=model_name, estimator=estimator)

        # labels
        train_y, val_y = self._get_labels()

        # pscores
        train_pscores, val_pscores = self._get_pscores(estimator=estimator)

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

    def _validate_params(self, model_name: str, estimator: str) -> None:
        """Validate model_name and estimator

        Args:
        - model_name (str): Model name (FM or MF)
        - estimator (str): Estimator name (IPS or Naive)

        Raises:
        - ValueError: model_name must be FM or MF
        - ValueError: estimator must be IPS or Naive
        """

        # params validation
        if model_name not in {"FM", "MF"}:
            self.logger.error(MODEL_NAME_ERROR_MESSAGE.format(model_name))
            raise ValueError(MODEL_NAME_ERROR_MESSAGE.format(model_name))
        if estimator not in {"Ideal", "IPS", "Naive"}:
            self.logger.error(ESTIMATOR_NAME_ERROR_MESSAGE.format(estimator))
            raise ValueError(ESTIMATOR_NAME_ERROR_MESSAGE.format(estimator))

    def _get_labels(self) -> Tuple[np.ndarray, ...]:
        """Get labels for training and evaluation

        Returns:
        - Tuple[np.ndarray]: labels for training and evaluation
        """

        labels = []
        for _df_name, _df in self.dataframes.items():
            if _df_name in {"train", "val"}:
                labels.append(_df["label"].values)

        return labels

    def _get_pscores(self, estimator: str) -> Tuple[np.ndarray, ...]:
        """Get pscores for training and evaluation

        Args:
            estimator (str): Estimator name (IPS or Naive)

        Returns:
            Tuple[np.ndarray, ...]: pscores for training and evaluation
        """

        col_name = "pscore" if estimator == "IPS" else "ones_pscore"
        pscores = []
        for _df_name, _df in self.dataframes.items():
            if _df_name in {"train", "val"}:
                pscores.append(_df[col_name].values ** self._params.pow_used)

        return pscores

    @property
    def val_df(self):
        """Validation dataframe"""
        return self.dataframes["val"]

    @property
    def val_evaluation_features(self):
        """Validation data for evaluation"""
        return dict(MF=self.datasets["MF"]["val"], FM=self.datasets["FM"]["val"])

    @property
    def test_df(self):
        """Test dataframe"""
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
            .agg({"label": "sum"})
            .values.reshape(-1)
        )
        item_populalities = item_populalities / item_populalities.max()

        return item_populalities
