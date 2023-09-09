import hydra
from hydra.core.config_store import ConfigStore
from conf.setting.default import ExperimentConfig
import numpy as np
import pandas as pd
from utils.dataloader import dataloader
from src.fm import FactorizationMachine as FM
from src.mf import ProbabilisticMatrixFactorization as PMF
from utils.evaluate import Evaluator

cs = ConfigStore.instance()
cs.store(name="config", group="setting", node=ExperimentConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig) -> None:
    datasets, pscores, clicks = dataloader(params=cfg.setting)

    result_df = pd.DataFrame()
    for model in {"FM", "PMF"}:
        dataset = datasets[model]
        evaluator = Evaluator(
            X=dataset.test,
            y_true=clicks["test"]["unbiased"],
            indices_per_user=None,
        )

        for estimator in {"Ideal", "IPS", "Naive"}:
            if estimator == "IPS":
                train_pscores = pscores.train
                val_pscores = pscores.val
            else:
                train_pscores = np.ones_like(pscores.train)
                val_pscores = np.ones_like(pscores.val)

            if estimator == "Ideal":
                train_y = clicks["train"]["unbiased"]
                val_y = clicks["val"]["unbiased"]
            else:
                train_y = clicks["train"]["biased"]
                val_y = clicks["val"]["biased"]

            test_y = clicks["test"]["unbiased"]

            if model == "FM":
                trains = tuple([dataset.train, train_y, train_pscores])
                vals = tuple([dataset.val, val_y, val_pscores])
                tests = tuple([dataset.test, test_y])

                model = FM(
                    n_epochs=cfg.setting.fm.n_epochs,
                    n_factors=cfg.setting.fm.n_factors,
                    n_features=dataset.train.shape[1],
                    scale=cfg.setting.fm.scale,
                    lr=cfg.setting.fm.lr,
                    batch_size=cfg.setting.fm.batch_size,
                    seed=cfg.setting.seed,
                )

            elif model == "PMF":
                train = np.concatenate(
                    [dataset.train, train_y[:, None]], axis=1
                )
                val = np.concatenate([dataset.val, val_y[:, None]], axis=1)
                tests = np.concatenate([dataset.test, test_y[:, None]], axis=1)

                trains = tuple([train, train_pscores])
                vals = tuple([val, val_pscores])

                model = PMF(
                    n_epochs=cfg.setting.mf.n_epochs,
                    n_factors=cfg.setting.mf.n_factors,
                    n_users=datasets["n_users"],
                    n_items=datasets["n_items"],
                    scale=cfg.setting.mf.scale,
                    lr=cfg.setting.mf.lr,
                    reg=cfg.setting.mf.reg,
                    batch_size=cfg.setting.mf.batch_size,
                    seed=cfg.setting.seed,
                )

            _, _, test_loss = model.fit(trains, vals, tests)
            recall_at_k, precision_at_k, dcg_at_k, _ = evaluator.evaluate(
                model
            )

            metrics = {
                "loss": test_loss,
                "recall": recall_at_k,
                "precision": precision_at_k,
                "dcg": dcg_at_k,
            }
            base_name = f"{model}_{estimator}"
            for metric, values in metrics.items():
                result_df[f"{base_name}_{metric}"] = [values]

    result_df.to_csv("./data/result.csv", index=False)


if __name__ == "__main__":
    main()
