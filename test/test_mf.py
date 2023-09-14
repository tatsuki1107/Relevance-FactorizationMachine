from test.test_fm import ModelTestBase
from src.mf import ProbabilisticMatrixFactorization as PMF
import numpy as np


class TestPMF(ModelTestBase):
    def setup_method(self):
        super().setup_method()

        # prepare data for IPS estimator
        dataset = self.datasets["PMF"]

        train_y = self.datasets["clicks"]["train"]["biased"]
        val_y = self.datasets["clicks"]["val"]["biased"]
        test_y = self.datasets["clicks"]["test"]["unbiased"]

        train_pscores = self.datasets["pscores"]["train"]
        val_pscores = self.datasets["pscores"]["val"]

        train = np.concatenate([dataset.train, train_y[:, None]], axis=1)
        val = np.concatenate([dataset.val, val_y[:, None]], axis=1)
        self.tests = np.concatenate([dataset.test, test_y[:, None]], axis=1)

        self.trains = tuple([train, train_pscores])
        self.vals = tuple([val, val_pscores])

        self.model = PMF(
            n_epochs=self.cfg.mf.n_epochs,
            n_factors=self.cfg.mf.n_factors,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            scale=self.cfg.mf.scale,
            lr=self.cfg.mf.lr,
            reg=self.cfg.mf.reg,
            batch_size=self.cfg.mf.batch_size,
            seed=self.cfg.seed,
        )

    def test_fit(self):
        losses = self.model.fit(self.trains, self.vals, self.tests)
        assert isinstance(losses, tuple)
