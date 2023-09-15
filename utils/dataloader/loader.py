from dataclasses import dataclass
from conf.config import ExperimentConfig
from utils.dataloader._click import ClickDataGenerator
from utils.dataloader._feature import FeatureGenerator
from utils.dataloader._kuairec import KuaiRecCSVLoader
from utils.dataloader._preparer import DatasetPreparer


@dataclass
class DataLoader:
    def load(self, params: ExperimentConfig) -> dict:
        """半人工データを生成する"""

        # small_matrix.csvのインタラクションデータを研究に用いる
        small_matrix_df = KuaiRecCSVLoader.create_interaction_df(
            params=params.tables.interaction
        )
        # 自然に観測されたbig_matrix上でのユーザーとアイテムの相対的な露出を用いて、
        # クリックデータを生成する
        click_generator = ClickDataGenerator(seed=params.seed)
        interaction_df = click_generator.generate_logdata_using_observed_data(
            interaction_df=small_matrix_df,
            params=params.logdata_propensity,
        )
        del small_matrix_df
        # interaction_dfに存在するユーザーとアイテムの特徴量を生成する
        feature_generator = FeatureGenerator(params=params.tables)
        features, interaction_df = feature_generator.generate(
            interaction_df=interaction_df
        )

        # FMとPMFを用いて学習、評価できるようにデータを整形する
        train_val_test_ratio = params.logdata_propensity.train_val_test_ratio
        preparer = DatasetPreparer()
        datasets = preparer.prepare_dataset(
            interaction_df=interaction_df,
            features=features,
            train_val_test_ratio=train_val_test_ratio,
        )
        return datasets
