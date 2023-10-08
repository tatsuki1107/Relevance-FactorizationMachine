from test.test_fm import ModelTestBase
from src.mf import LogisticMatrixFactorization as MF


class TestMF(ModelTestBase):
    def setup_method(self):
        super().setup_method()

        # prepare data for IPS estimator
        self.train, self.val, _ = self.loader.load(
            model_name="MF", estimator="IPS"
        )

        self.model = MF(
            n_epochs=self.cfg.model.PMF.n_epochs[0],
            n_factors=self.cfg.model.PMF.n_factors[0],
            n_users=self.loader.n_users,
            n_items=self.loader.n_items,
            lr=self.cfg.model.PMF.lr[0],
            reg=self.cfg.model.PMF.reg[0],
            batch_size=self.cfg.model.PMF.batch_size[0],
            seed=self.cfg.seed,
        )

    def test_fit(self):
        losses = self.model.fit(self.train, self.val)
        assert isinstance(losses, tuple)
