# Internal modules imports
from test.test_fm import ModelTestBase
from src.mf import LogisticMatrixFactorization as MF


class TestMF(ModelTestBase):
    def setup_method(self):
        super().setup_method()

        # prepare data for IPS estimator
        self.train, self.val, _ = self.loader.load(
            model_name="MF", estimator="IPS"
        )
        
        model_config = self.cfg.model_param_range.MF
        self.model = MF(
            n_epochs=model_config.n_epochs.min,
            n_factors=model_config.n_factors.min,
            n_users=self.loader.n_users,
            n_items=self.loader.n_items,
            lr=model_config.lr.min,
            reg=model_config.reg.min,
            batch_size=model_config.batch_size.min,
            seed=self.cfg.seed,
        )

    def test_fit(self):
        losses = self.model.fit(self.train, self.val)
        assert isinstance(losses, tuple)
