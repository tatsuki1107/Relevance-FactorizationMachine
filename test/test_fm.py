from abc import ABC
from hydra import initialize, compose
from conf.config import ExperimentConfig
from utils.loader import DataLoader
from src.fm import FactorizationMachine as FM


class ModelTestBase(ABC):
    def setup_method(self):
        initialize(config_path="../conf", version_base="1.3")
        self.cfg: ExperimentConfig = compose(config_name="config")
        loader = DataLoader()
        self.datasets = loader.load(self.cfg)


class TestFM(ModelTestBase):
    def setup_method(self):
        super().setup_method()

        # prepare data for IPS estimator
        dataset = self.datasets["FM"]

        train_y = self.datasets["clicks"]["train"]["biased"]
        val_y = self.datasets["clicks"]["val"]["biased"]
        test_y = self.datasets["clicks"]["test"]["unbiased"]

        train_pscores = self.datasets["pscores"]["train"]
        val_pscores = self.datasets["pscores"]["val"]

        self.trains = tuple([dataset.train, train_y, train_pscores])
        self.vals = tuple([dataset.val, val_y, val_pscores])
        self.tests = tuple([dataset.test, test_y])

        self.model = FM(
            n_epochs=self.cfg.fm.n_epochs,
            n_factors=self.cfg.fm.n_factors,
            n_features=dataset.train.shape[1],
            scale=self.cfg.fm.scale,
            lr=self.cfg.fm.lr,
            batch_size=self.cfg.fm.batch_size,
            seed=self.cfg.seed,
        )

    def test_fit(self):
        losses = self.model.fit(self.trains, self.vals, self.tests)
        assert isinstance(losses, tuple)
