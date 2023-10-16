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
            n_epochs=self.cfg.model_param_range.MF.n_epochs.min,
            n_factors=self.cfg.model_param_range.MF.n_factors.min,
            n_users=self.loader.n_users,
            n_items=self.loader.n_items,
            lr=self.cfg.model_param_range.MF.lr.min,
            reg=self.cfg.model_param_range.MF.reg.min,
            batch_size=self.cfg.model_param_range.MF.batch_size.min,
            seed=self.cfg.seed,
        )

    def test_fit(self):
        losses = self.model.fit(self.train, self.val)
        assert isinstance(losses, tuple)
