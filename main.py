# Standard library imports
from logging import getLogger
from pathlib import Path
import json

# Third-party library imports
import hydra
from hydra.core.config_store import ConfigStore
import pandas as pd

# Internal modules imports
from conf.config import ExperimentConfig
from utils.search_params import random_search
from utils.dataloader.loader import DataLoader
from utils.evaluate import TestEvaluator
from utils.plot import Visualizer, plot_loss_curve
from src.fm import FactorizationMachines as FM
from src.mf import LogisticMatrixFactorization as MF


LOG_PATH = Path("./logs/result")
PARAMS_PATH = Path("./data/best_params")

K = [i for i in range(1, 6)]
METRICS = {
    "DCG",
    "Recall",
    "MAP",
    "ME",
}


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

    LOG_PATH.mkdir(exist_ok=True, parents=True)

    logger.info("start data loading...")
    dataloader = DataLoader(cfg, logger)
    user2data_indices = dataloader.test_user2data_indices

    logger.info("data loading is done.")

    if cfg.is_search_params:
        random_search(
            model_config=cfg.model_param_range,
            seed=cfg.seed,
            dataloader=dataloader,
            logger=logger,
        )

    logger.info("start experiment...")

    metric_df = pd.DataFrame()
    for model_name in ["FM", "MF"]:
        for estimator in ["Ideal", "IPS", "Naive"]:
            base_name = f"{model_name}_{estimator}"

            train, val, test = dataloader.load(
                model_name=model_name, estimator=estimator
            )

            with open(
                PARAMS_PATH / f"{model_name}_{estimator}_best_param.json",
                "r",
            ) as f:
                model_params = json.load(f)

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

            train_loss, val_loss = model.fit(train, val)
            plot_loss_curve(train_loss, val_loss, base_name)

            for frequency, user2indices in user2data_indices.items():
                evaluator = TestEvaluator(
                    _seed=cfg.seed,
                    X=test[0],
                    y_true=test[1],
                    indices_per_user=user2indices,
                    used_metrics=METRICS,
                    K=K,
                )

                results = evaluator.evaluate(model, pscores=test[2])
                for metric_name, values in results.items():
                    metric_df[
                        f"{base_name}_{frequency}_{metric_name}@K"
                    ] = values

            logger.info(f"{base_name} is done.")

    # random baseline
    model_name = "Random"
    test_data = dataloader.test_data_for_random_policy
    for frequency, user2indices in user2data_indices.items():
        evaluator = TestEvaluator(
            _seed=cfg.seed,
            X=None,
            y_true=test_data["y_true"],
            indices_per_user=user2indices,
            used_metrics=METRICS,
            K=K,
        )

        results = evaluator.evaluate(
            model=model_name, pscores=test_data["pscore"]
        )
        for metric_name, values in results.items():
            metric_df[f"Random_{frequency}_{metric_name}@K"] = values

    logger.info(f"{model_name} is done.")

    metric_df.to_csv(LOG_PATH / "metric.csv", index=False)

    # visualize and save the results
    Visualizer(
        result_df=metric_df,
        K=K,
    )


if __name__ == "__main__":
    main()
