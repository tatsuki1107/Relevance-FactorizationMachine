# Standard library imports
from abc import ABC
from logging import getLogger
from collections import defaultdict

# Third-party library imports
from hydra import initialize, compose

# Internal modules imports
from conf.config import ExperimentConfig
from utils.evaluate import TestEvaluator
from utils.dataloader.loader import DataLoader
from src.fm import FactorizationMachines as FM


class ModelTestBase(ABC):
    def setup_method(self):
        logger = getLogger(__name__)
        initialize(config_path="../conf", version_base="1.3")
        self.cfg: ExperimentConfig = compose(config_name="config")
        self.loader = DataLoader(self.cfg, logger)


class TestFM(ModelTestBase):
    def setup_method(self):
        super().setup_method()

        # prepare data for IPS estimator
        self.train, self.val, self.test = self.loader.load(
            model_name="FM", estimator="Naive"
        )

        model_config = self.cfg.model_param_range.FM
        self.model = FM(
            n_epochs=model_config.n_epochs.min,
            n_factors=model_config.n_factors.min,
            n_features=self.train[0].shape[1],
            lr=model_config.lr.min,
            batch_size=model_config.batch_size.min,
            seed=self.cfg.seed,
        )

        user2data_indices = self.loader.test_user2data_indices["all"]

        self.evaluator = TestEvaluator(
            _seed=self.cfg.seed,
            X=self.test[0],
            y_true=self.test[1],
            indices_per_user=user2data_indices,
            used_metrics={"DCG", "MAP", "Recall"},
        )

    def test_fit(self):
        _, batch_y, _ = self.model._get_batch_data(
            train_X=self.train[0],
            train_y=self.train[1],
            train_pscores=self.train[2],
        )
        assert len(batch_y[0]) == self.cfg.model_param_range.FM.batch_size.min
        train_loss, _ = self.model.fit(self.train, self.val)
        assert isinstance(train_loss, list)

        # evaluate
        results = self.evaluator.evaluate(self.model)
        assert isinstance(results, defaultdict)
