from logging import getLogger
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import ExperimentConfig
import numpy as np
import pandas as pd
from utils.loader import DataLoader
from src.fm import FactorizationMachine as FM
from src.mf import ProbabilisticMatrixFactorization as PMF
from utils.evaluate import Evaluator

cs = ConfigStore.instance()
cs.store(name="setting", node=ExperimentConfig)

logger = getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig) -> None:
    log_path = Path("./data/result")
    log_path.mkdir(exist_ok=True, parents=True)

    logger.info("start data loading...")

    loader = DataLoader()
    datasets = loader.load(cfg)
    clicks = datasets["clicks"]
    pscores = datasets["pscores"]
    test_user2indices = datasets["test_user2indices"]

    logger.info("data loading is done.")

    metric_df = pd.DataFrame()
    for model_name in {"FM", "PMF"}:
        dataset = datasets[model_name]

        loss_df = pd.DataFrame()

        for estimator in {"Ideal", "IPS", "Naive"}:
            if estimator == "IPS":
                train_pscores = pscores["train"]
                val_pscores = pscores["val"]
            else:
                train_pscores = np.ones_like(pscores["train"])
                val_pscores = np.ones_like(pscores["val"])

            if estimator == "Ideal":
                train_y = clicks["train"]["unbiased"]
                val_y = clicks["val"]["unbiased"]
            else:
                train_y = clicks["train"]["biased"]
                val_y = clicks["val"]["biased"]

            test_y = clicks["test"]["unbiased"]

            if model_name == "FM":
                trains = tuple([dataset.train, train_y, train_pscores])
                vals = tuple([dataset.val, val_y, val_pscores])
                tests = tuple([dataset.test, test_y])

                model = FM(
                    n_epochs=cfg.fm.n_epochs,
                    n_factors=cfg.fm.n_factors,
                    n_features=dataset.train.shape[1],
                    scale=cfg.fm.scale,
                    lr=cfg.fm.lr,
                    batch_size=cfg.fm.batch_size,
                    seed=cfg.seed,
                )

            elif model_name == "PMF":
                train = np.concatenate(
                    [dataset.train, train_y[:, None]], axis=1
                )
                val = np.concatenate([dataset.val, val_y[:, None]], axis=1)
                tests = np.concatenate([dataset.test, test_y[:, None]], axis=1)

                trains = tuple([train, train_pscores])
                vals = tuple([val, val_pscores])

                model = PMF(
                    n_epochs=cfg.mf.n_epochs,
                    n_factors=cfg.mf.n_factors,
                    n_users=dataset.n_users,
                    n_items=dataset.n_items,
                    scale=cfg.mf.scale,
                    lr=cfg.mf.lr,
                    reg=cfg.mf.reg,
                    batch_size=cfg.mf.batch_size,
                    seed=cfg.seed,
                )

            _, _, test_loss = model.fit(trains, vals, tests)

            for frequency, user2_indices in test_user2indices.items():
                evaluator = Evaluator(
                    X=dataset.test,
                    y_true=test_y,
                    indices_per_user=user2_indices,
                )

                recall_at_k, precision_at_k, dcg_at_k, _ = evaluator.evaluate(
                    model
                )

                metrics = {
                    f"{frequency}_Recall@K": recall_at_k,
                    f"{frequency}_Precision@K": precision_at_k,
                    f"{frequency}_DCG@K": dcg_at_k,
                }
                base_name = f"{model_name}_{estimator}"
                for metric, values in metrics.items():
                    metric_df[f"{base_name}_{metric}"] = values

            loss_df[f"{base_name}_loss"] = test_loss

            logger.info(f"{base_name} is done.")

        loss_df.to_csv(log_path / f"{model_name}_loss.csv", index=False)

    metric_df.to_csv(log_path / "metric.csv", index=False)


if __name__ == "__main__":
    main()
