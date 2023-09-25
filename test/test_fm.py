from abc import ABC
from collections import defaultdict
from hydra import initialize, compose
from conf.config import ExperimentConfig
from utils.evaluate import Evaluator
from utils.dataloader.loader import DataLoader
from src.fm import FactorizationMachine as FM


class ModelTestBase(ABC):
    def setup_method(self):
        initialize(config_path="../conf", version_base="1.3")
        self.cfg: ExperimentConfig = compose(config_name="config")
        self.loader = DataLoader(self.cfg)


class TestFM(ModelTestBase):
    def setup_method(self):
        super().setup_method()

        # prepare data for IPS estimator
        self.train, self.val, self.test = self.loader.load(
            model_name="FM", estimator="IPS"
        )

        self.model = FM(
            n_epochs=self.cfg.model.FM.n_epochs[0],
            n_factors=self.cfg.model.FM.n_factors[0],
            n_features=self.train[0].shape[1],
            scale=self.cfg.model.FM.scale[0],
            lr=self.cfg.model.FM.lr[0],
            batch_size=self.cfg.model.FM.batch_size[0],
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
