# Standard library imports
from typing import Tuple
from logging import Logger
from dataclasses import dataclass
from collections import defaultdict

# Third-party library imports
import numpy as np

# Internal modules imports
from conf.config import ExperimentConfig
from utils.dataloader._click import SemiSyntheticLogDataGenerator
from utils.dataloader._feature import FeatureGenerator
from utils.dataloader._kuairec import KuaiRecCSVLoader
from utils.dataloader._preparer import DatasetPreparer
from utils.dataloader.base import BaseLoader

MODEL_NAME_ERROR_MESSAGE = "model_name must be FM or MF. model_name: '{}'"
ESTIMATOR_NAME_ERROR_MESSAGE = (
    "estimator must be Ideal, IPS or Naive. estimator: '{}'"
)


@dataclass
class DataLoader(BaseLoader):
    """学習、評価に用いるデータを生成する

    Args:
    - _params (ExperimentConfig): 実験設定のパラメータ(read only)
    - logger (Logger): Loggerクラスのインスタンス
    """

    _params: ExperimentConfig
    logger: Logger

    def __post_init__(self) -> None:
        """半人工データを生成する"""

        # small_matrix.csvのインタラクションデータを研究に用いる
        small_matrix_df = KuaiRecCSVLoader.create_small_matrix_df(
            _params=self._params.tables.interaction,
            logger=self.logger,
        )
        self.logger.info("created small_matrix_df")

        # 半人工ログデータを生成する
        logdata_generator = SemiSyntheticLogDataGenerator(
            _seed=self._params.seed,
            _params=self._params.data_logging_settings,
            logger=self.logger,
        )
        interaction_df = logdata_generator.load(
            interaction_df=small_matrix_df,
        )
        self.logger.info("created interaction_df")

        # interaction_dfに存在するユーザーとアイテムの特徴量を生成する
        feature_generator = FeatureGenerator(
            _params=self._params.tables,
            logger=self.logger,
        )
        features, interaction_df = feature_generator.load(
            interaction_df=interaction_df
        )
        self.logger.info("created features")

        # FMとMFを用いて学習、評価できるようにデータを整形する
        preparer = DatasetPreparer(_seed=self._params.seed)
        self.datasets = preparer.load(
            interaction_df=interaction_df,
            features=features,
        )
        self.logger.info("prepared train and evaluation datasets")

        self.n_users = self.datasets["MF"]["n_users"]
        self.n_items = self.datasets["MF"]["n_items"]

    def load(self, model_name: str, estimator: str) -> Tuple[list]:
        """モデルと推定量ごとのtrain, val, testデータを返す

        Args:
        - model_name (str): モデルの名前。FMまたは、MF
        - estimator (str): 推定量の名前。Ideal, IPSまたは、Naive

        Raises:
        - ValueError: model_nameがFMまたは、MFでない場合
        - ValueError: estimatorがIdeal, IPSまたは、Naiveでない場合

        Returns:
        - (tuple): train, val, testデータ
        """

        # params validation
        if model_name not in {"FM", "MF"}:
            self.logger.error(MODEL_NAME_ERROR_MESSAGE.format(model_name))
            raise ValueError(MODEL_NAME_ERROR_MESSAGE.format(model_name))
        if estimator not in {"Ideal", "IPS", "Naive"}:
            self.logger.error(ESTIMATOR_NAME_ERROR_MESSAGE.format(estimator))
            raise ValueError(ESTIMATOR_NAME_ERROR_MESSAGE.format(estimator))

        # target variable
        if estimator == "Ideal":
            train_y = self.datasets["relevances"]["train"]
            val_y = self.datasets["relevances"]["val"]
        else:
            train_y = self.datasets["clicks"]["train"]
            val_y = self.datasets["clicks"]["val"]

        test_y = self.datasets["clicks"]["test"]

        # features
        features = self.datasets[model_name]

        # estimated exposure
        if estimator == "IPS":
            train_pscores = self.datasets["pscores"]["train"]
            val_pscores = self.datasets["pscores"]["val"]
        else:
            train_pscores = np.ones_like(train_y)
            val_pscores = np.ones_like(val_y)

        # negative sampling
        if estimator in {"IPS", "Naive"}:
            sampled_train_indices = self.datasets["sampled_train_indices"]
            sampled_train_X = features["train"][sampled_train_indices]
            sampled_train_y = train_y[sampled_train_indices]
            sampled_train_pscores = train_pscores[sampled_train_indices]
            train = [sampled_train_X, sampled_train_y, sampled_train_pscores]
        else:
            train = [features["train"], train_y, train_pscores]

        val = [features["val"], val_y, val_pscores]
        test = [features["test"], test_y]

        return train, val, test

    @property
    def val_user2data_indices(self) -> dict:
        """validationデータのユーザーごとのデータインデックスを返す"""
        return self.datasets["user2data_indices"]["val"]["all"]

    @property
    def val_data_for_random_policy(self) -> defaultdict:
        """ランダムベースラインの評価に用いるvalidationデータを返す"""

        estimators = ["Ideal", "IPS", "Naive"]
        val_data = defaultdict(dict)
        for estimator in estimators:
            if estimator == "Ideal":
                val_data[estimator]["y_true"] = self.datasets["relevances"][
                    "val"
                ]
                val_data[estimator]["pscore"] = None
            elif estimator == "IPS":
                val_data[estimator]["y_true"] = self.datasets["clicks"]["val"]
                val_data[estimator]["pscore"] = self.datasets["pscores"]["val"]
            else:
                val_data[estimator]["y_true"] = self.datasets["clicks"]["val"]
                val_data[estimator]["pscore"] = None

        return val_data

    @property
    def test_user2data_indices(self) -> dict:
        """testデータのユーザーごとのデータインデックスを返す"""
        return self.datasets["user2data_indices"]["test"]

    @property
    def test_y(self) -> np.ndarray:
        """testデータのターゲット変数を返す"""
        return self.datasets["clicks"]["test"]
