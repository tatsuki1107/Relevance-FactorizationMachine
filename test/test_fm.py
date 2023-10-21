# Standard library imports
from abc import ABC
from logging import getLogger
from collections import defaultdict

# Third-party library imports
from hydra import initialize, compose

# Internal modules imports
from conf.config import ExperimentConfig
from utils.evaluate import Evaluator
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
            model_name="FM", estimator="IPS"
        )

        self.model = FM(
            n_epochs=self.cfg.model_param_range.FM.n_epochs.min,
            n_factors=self.cfg.model_param_range.FM.n_factors.min,
            n_features=self.train[0].shape[1],
            lr=self.cfg.model_param_range.FM.lr.min,
            batch_size=self.cfg.model_param_range.FM.batch_size.min,
            seed=self.cfg.seed,
        )

        user2data_indices = self.loader.test_user2data_indices["all"]

        self.evaluator = Evaluator(
            X=self.test[0],
            y_true=self.test[1],
            indices_per_user=user2data_indices,
            used_metrics={"DCG", "Precision", "Recall"},
        )

    def test_fit(self):
        losses = self.model.fit(self.train, self.val)
        assert isinstance(losses, tuple)

        # evaluate
        results = self.evaluator.evaluate(self.model)
        assert isinstance(results, defaultdict)
