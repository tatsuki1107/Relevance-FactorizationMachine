# Standard library imports
from logging import getLogger

# Third-party library imports
from hydra import initialize, compose
import numpy as np

# Internal modules imports
from utils.dataloader.loader import DataLoader


class TestDataLoader:
    def setup_method(self):
        initialize(config_path="../conf", version_base="1.3")
        self.cfg = compose(config_name="config")

    def test_dataloader(self):
        logger = getLogger(__name__)
        loader = DataLoader(self.cfg, logger)
        datasets = loader.load(model_name="MF", estimator="IPS")
        sampled_train_y = datasets[0][1]
        label_counts = np.unique(sampled_train_y, return_counts=True)[1]
        assert label_counts[0] == label_counts[1]
