from hydra import initialize, compose
from utils.dataloader.loader import DataLoader


class TestDataLoader:
    def setup_method(self):
        initialize(config_path="../conf", version_base="1.3")
        self.cfg = compose(config_name="config")

    def test_dataloader(self):
        loader = DataLoader()
        datasets = loader.load(self.cfg)

        assert isinstance(datasets, dict)
