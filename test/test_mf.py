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
            n_epochs=self.cfg.model.MF.n_epochs[0],
            n_factors=self.cfg.model.MF.n_factors[0],
            n_users=self.loader.n_users,
            n_items=self.loader.n_items,
            lr=self.cfg.model.MF.lr[0],
            reg=self.cfg.model.MF.reg[0],
            batch_size=self.cfg.model.MF.batch_size[0],
            seed=self.cfg.seed,
        )

    def test_fit(self):
        losses = self.model.fit(self.train, self.val)
        assert isinstance(losses, tuple)
