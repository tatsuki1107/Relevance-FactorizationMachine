from hydra import initialize, compose
from utils.dataloader.loader import DataLoader


class TestDataLoader:
    def setup_method(self):
        initialize(config_path="../conf", version_base="1.3")
        self.cfg = compose(config_name="config")

    def test_dataloader(self):
        loader = DataLoader(self.cfg)
        datasets = loader.load(model_name="FM", estimator="IPS")

        assert isinstance(datasets, tuple)
        assert isinstance(loader.test_user2data_indices, dict)
        assert isinstance(loader.val_user2data_indices, list)
