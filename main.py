from logging import getLogger
from pathlib import Path
import json
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import ExperimentConfig
import pandas as pd
import numpy as np
from utils.search_params import random_search
from utils.dataloader.loader import DataLoader
from src.fm import FactorizationMachines as FM
from src.mf import LogisticMatrixFactorization as MF
from utils.evaluate import Evaluator
from utils.plot import Visualizer

cs = ConfigStore.instance()
cs.store(name="setting", node=ExperimentConfig)

logger = getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig) -> None:
    """実験を実行する関数. 実験の詳細は、short_paper.mdを参照
        FM, MFモデルにそれぞれの損失(Ideal, IPS, Naive)を適用させ、testデータでのランキング性能を評価する
    Args:
    - cfg (ExperimentConfig): 実験設定のパラメータ
    """

    log_path = Path("./logs/result")
    log_path.mkdir(exist_ok=True, parents=True)

    params_path = Path("./data/best_params")

    logger.info("start data loading...")
    dataloader = DataLoader(cfg, logger)
    user2data_indices = dataloader.test_user2data_indices

    logger.info("data loading is done.")

    if cfg.is_search_params:
        random_search(
            model_config=cfg.model,
            seed=cfg.seed,
            dataloader=dataloader,
            logger=logger,
        )

    logger.info("start experiment...")

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
                    lr=model_params["lr"],
                    batch_size=model_params["batch_size"],
                    seed=cfg.seed,
                )

            elif model_name == "MF":
                model = MF(
                    n_epochs=model_params["n_epochs"],
                    n_factors=model_params["n_factors"],
                    n_users=dataloader.n_users,
                    n_items=dataloader.n_items,
                    lr=model_params["lr"],
                    reg=model_params["reg"],
                    batch_size=model_params["batch_size"],
                    seed=cfg.seed,
                )

            _, _ = model.fit(train, val)

            pred_scores = model.predict(test[0])
            logloss = model._cross_entropy_loss(
                y_trues=test[1],
                y_scores=pred_scores,
                pscores=np.ones_like(test[1]),
            )
            logloss_df[base_name] = [logloss]

            print(
                f"{base_name}: min: {pred_scores.min()},"
                + f"max: {pred_scores.max()}, mean: {pred_scores.mean()}"
                + f"std: {pred_scores.std()}"
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

    Visualizer(
        result_df=metric_df,
        K=K,
        metrics=used_metrics,
    )


if __name__ == "__main__":
    main()
