import numpy as np
from dataclasses import dataclass
from conf.config import ExperimentConfig
from utils.dataloader._click import ClickDataGenerator
from utils.dataloader._feature import FeatureGenerator
from utils.dataloader._kuairec import KuaiRecCSVLoader
from utils.dataloader._preparer import DatasetPreparer


@dataclass
class DataLoader:
    params: ExperimentConfig

    def __post_init__(self) -> None:
        """半人工データを生成する"""

        # small_matrix.csvのインタラクションデータを研究に用いる
        small_matrix_df = KuaiRecCSVLoader.create_interaction_df(
            params=self.params.tables.interaction
        )
        # 自然に観測されたbig_matrix上でのユーザーとアイテムの相対的な露出を用いて、
        # クリックデータを生成する
        click_generator = ClickDataGenerator(seed=self.params.seed)
        interaction_df = click_generator.generate_logdata_using_observed_data(
            interaction_df=small_matrix_df,
            params=self.params.logdata_propensity,
        )
        del small_matrix_df
        # interaction_dfに存在するユーザーとアイテムの特徴量を生成する
        feature_generator = FeatureGenerator(params=self.params.tables)
        features, interaction_df = feature_generator.generate(
            interaction_df=interaction_df
        )

        # FMとPMFを用いて学習、評価できるようにデータを整形する
        train_val_test_ratio = (
            self.params.logdata_propensity.train_val_test_ratio
        )
        preparer = DatasetPreparer()
        self.datasets = preparer.prepare_dataset(
            interaction_df=interaction_df,
            features=features,
            train_val_test_ratio=train_val_test_ratio,
        )

    def load(self, model_name: str, estimator: str) -> tuple:
        """モデルと推定量ごとのtrain, val, testデータを返す"""
        # params validation
        if model_name not in {"FM", "PMF"}:
            raise ValueError(
                "model_name must be FM or PMF. " + f"model_name: {model_name}"
            )
        if estimator not in {"Ideal", "IPS", "Naive"}:
            raise ValueError(
                "estimator must be Ideal, IPS or Naive."
                + f" estimator: {estimator}"
            )

        # features
        features = self.datasets[model_name]

        # clicks
        clicks = self.datasets["clicks"]
        if estimator == "Ideal":
            train_y = clicks["train"]["unbiased"]
            val_y = clicks["val"]["unbiased"]
        else:
            train_y = clicks["train"]["biased"]
            val_y = clicks["val"]["biased"]

        test_y = clicks["test"]["unbiased"]

        # exposure
        pscores = self.datasets["pscores"]
        if estimator == "IPS":
            train_pscores = pscores["train"]
            val_pscores = pscores["val"]
        else:
            train_pscores = np.ones_like(pscores["train"])
            val_pscores = np.ones_like(pscores["val"])

        # train val test
        if model_name == "FM":
            train = tuple([features.train, train_y, train_pscores])
            val = tuple([features.val, val_y, val_pscores])
            test = tuple([features.test, test_y])
        elif model_name == "PMF":
            _train = np.concatenate([features.train, train_y[:, None]], axis=1)
            _val = np.concatenate([features.val, val_y[:, None]], axis=1)
            test = np.concatenate([features.test, test_y[:, None]], axis=1)

            train = tuple([_train, train_pscores])
            val = tuple([_val, val_pscores])

        # test_user2indices
        test_user2indices = self.datasets["test_user2indices"]

        return train, val, test, test_user2indices
