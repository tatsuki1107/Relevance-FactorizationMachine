from logging import getLogger
from pathlib import Path
import json
import hydra

# import joblib
from hydra.core.config_store import ConfigStore
from conf.config import ExperimentConfig
import pandas as pd
import numpy as np

from utils.dataloader.loader import DataLoader
from src.fm import FactorizationMachine as FM
from src.mf import LogisticMatrixFactorization as MF
from utils.evaluate import Evaluator

cs = ConfigStore.instance()
cs.store(name="setting", node=ExperimentConfig)

logger = getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig) -> None:
    log_path = Path("./data/sample_result")
    log_path.mkdir(exist_ok=True, parents=True)

    params_path = Path("./data/uniform_init_best_params")

    logger.info("start data loading...")
    # data_path = Path("./data/best_params")
    # with open(data_path / "dataloader.pkl", "rb") as f:
    #    dataloader = joblib.load(f)
    dataloader = DataLoader(cfg)
    user2data_indices = dataloader.test_user2data_indices

    logger.info("data loading is done.")
    K = [i for i in range(1, 11)]
    metric_df = pd.DataFrame()
    logloss_df = pd.DataFrame()
    used_metrics = {"DCG", "Recall", "MAP"}
    for model_name in ["FM", "MF"]:
        for estimator in ["Ideal", "IPS", "Naive"]:
            base_name = f"{model_name}_{estimator}"

            train, val, test = dataloader.load(
                model_name=model_name, estimator=estimator
            )

            with open(
                params_path / f"{model_name}_{estimator}_best_param.json",
                "r",
            ) as f:
                model_params = json.load(f)

            if estimator == "IPS":
                # pscore clipping
                train[2] = np.maximum(train[2], model_params["clipping"])
                val[2] = np.maximum(val[2], model_params["clipping"])

            if model_name == "FM":
                model = FM(
                    n_epochs=model_params["n_epochs"],
                    n_factors=model_params["n_factors"],
                    n_features=train[0].shape[1],
                    scale=model_params["scale"],
                    lr=model_params["lr"],
                    batch_size=model_params["batch_size"],
                    seed=cfg.seed,
                )
                _, _ = model.fit(train, val)

                logloss = model._cross_entropy_loss(
                    X=test[0],
                    y=test[1],
                    pscores=np.ones_like(test[1]),
                )
                logloss_df[base_name] = [logloss]
                pred_y = model.predict(test[0])

            elif model_name == "MF":
                model = MF(
                    n_epochs=model_params["n_epochs"],
                    n_factors=model_params["n_factors"],
                    n_users=dataloader.n_users,
                    n_items=dataloader.n_items,
                    scale=model_params["scale"],
                    lr=model_params["lr"],
                    reg=model_params["reg"],
                    batch_size=model_params["batch_size"],
                    seed=cfg.seed,
                )

                _, _ = model.fit(train, val)

                logloss = model._cross_entropy_loss(
                    user_ids=test[0][:, 0],
                    item_ids=test[0][:, 1],
                    clicks=test[1],
                    pscores=np.ones_like(test[1]),
                )
                logloss_df[base_name] = [logloss]
                pred_y = model.predict(test[0][:, 0], test[0][:, 1])
            print(
                f"{base_name}: min: {pred_y.min()},"
                + f"max: {pred_y.max()}, mean: {pred_y.mean()}"
                + f"std: {pred_y.std()}"
            )

            for frequency, user2indices in user2data_indices.items():
                evaluator = Evaluator(
                    X=test[0],
                    y_true=test[1],
                    indices_per_user=user2indices,
                    used_metrics=used_metrics,
                    K=K,
                    thetahold=None,
                )

                results = evaluator.evaluate(model)
                for metric_name, values in results.items():
                    metric_df[
                        f"{base_name}_{frequency}_{metric_name}@K"
                    ] = values

            logger.info(f"{base_name} is done.")

    # random baseline
    model_name = "Random"
    test_y = dataloader.test_y
    for frequency, user2indices in user2data_indices.items():
        evaluator = Evaluator(
            X=None,
            y_true=test_y,
            indices_per_user=user2indices,
            used_metrics=used_metrics,
            K=K,
            thetahold=None,
        )

        results = evaluator.evaluate(model=model_name)
        for metric_name, values in results.items():
            metric_df[f"Random_{frequency}_{metric_name}@K"] = values

    logger.info(f"{model_name} is done.")

    metric_df.to_csv(log_path / "metric.csv", index=False)
    logloss_df.to_csv(log_path / "logloss.csv", index=False)


if __name__ == "__main__":
    main()
