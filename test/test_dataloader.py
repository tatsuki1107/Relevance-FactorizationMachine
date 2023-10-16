from hydra import initialize, compose
from logging import getLogger
from utils.dataloader.loader import DataLoader
import numpy as np


class TestDataLoader:
    def setup_method(self):
        initialize(config_path="../conf", version_base="1.3")
        self.cfg = compose(config_name="config")

    def test_dataloader(self):
        logger = getLogger(__name__)
        loader = DataLoader(self.cfg, logger)
        datasets = loader.load(model_name="FM", estimator="IPS")

        assert isinstance(datasets, tuple)
        assert isinstance(datasets[1][2], np.ndarray)
        assert isinstance(loader.test_user2data_indices, dict)
        assert isinstance(loader.val_user2data_indices, list)
