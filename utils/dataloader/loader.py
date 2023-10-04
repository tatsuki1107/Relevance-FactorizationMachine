import numpy as np
from dataclasses import dataclass
from conf.config import ExperimentConfig
from utils.dataloader._click import SemiSyntheticLogDataGenerator
from utils.dataloader._feature import FeatureGenerator
from utils.dataloader._kuairec import KuaiRecCSVLoader
from utils.dataloader._preparer import DatasetPreparer
from utils.dataloader.base import BaseLoader


@dataclass
class DataLoader(BaseLoader):
    _params: ExperimentConfig

    def __post_init__(self) -> None:
        """半人工データを生成する"""

        # small_matrix.csvのインタラクションデータを研究に用いる
        small_matrix_df = KuaiRecCSVLoader.create_interaction_df(
            _params=self._params.tables.interaction
        )
        # 自然に観測されたbig_matrix上でのユーザーとアイテムの相対的な露出を用いて、
        # クリックデータを生成する
        logdata_generator = SemiSyntheticLogDataGenerator(
            _seed=self._params.seed,
            _params=self._params.logdata_propensity,
        )
        interaction_df = logdata_generator.load(
            interaction_df=small_matrix_df,
        )

        del small_matrix_df
        # interaction_dfに存在するユーザーとアイテムの特徴量を生成する
        feature_generator = FeatureGenerator(_params=self._params.tables)
        features, interaction_df = feature_generator.load(
            interaction_df=interaction_df
        )

        # FMとPMFを用いて学習、評価できるようにデータを整形する
        preparer = DatasetPreparer(_seed=self._params.seed)
        self.datasets = preparer.load(
            interaction_df=interaction_df,
            features=features,
        )

        self.n_users = self.datasets["MF"].n_users
        self.n_items = self.datasets["MF"].n_items

    def load(self, model_name: str, estimator: str) -> tuple:
        """モデルと推定量ごとのtrain, val, testデータを返す"""
        # params validation
        if model_name not in {"FM", "MF"}:
            raise ValueError(
                "model_name must be FM or MF. " + f"model_name: {model_name}"
            )
        if estimator not in {"Ideal", "IPS", "Naive"}:
            raise ValueError(
                "estimator must be Ideal, IPS or Naive."
                + f" estimator: {estimator}"
            )

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

        # train val test
        train = [features.train, train_y, train_pscores]
        val = [features.val, val_y, val_pscores]
        test = [features.test, test_y]

        return train, val, test

    @property
    def val_user2data_indices(self) -> dict:
        return self.datasets["user2data_indices"]["val"]["all"]

    @property
    def test_user2data_indices(self) -> dict:
        return self.datasets["user2data_indices"]["test"]

    @property
    def test_y(self) -> np.ndarray:
        return self.datasets["clicks"]["test"]
