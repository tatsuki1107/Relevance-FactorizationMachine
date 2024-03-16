# Standard library imports
from logging import getLogger
from pathlib import Path
import json

# Third-party library imports
import hydra
from omegaconf import OmegaConf

from hydra.core.config_store import ConfigStore
import pandas as pd
import numpy as np

# Internal modules imports
from conf.kuairec import Config
from utils.search_params import search_params
from utils.dataloader.kuairec.loader import DataLoader
from utils.evaluate import TestEvaluator
from utils.plot import Visualizer, plot_populality
from src.fm import FactorizationMachines as FM
from src.mf import LogisticMatrixFactorization as MF

# recommendation position
K = [1, 3, 5, 7, 9]
# quantitave metric
QUANTITATIVE_METRIC = "DCG"
# qualitative metric
QUALITATIVE_METRIC = "CatalogCoverage"
# used estimators
ESTIMATORS = ["IPS", "Naive"]
# used algorithms
MODELS = ["FM", "MF"]

cs = ConfigStore.instance()
cs.store(name="setting", node=Config)

logger = getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    """Function to run the experiment. See short_paper.md for details of the experiment.
    Apply each loss (IPS, Naive) to FM and MF models,
    and evaluate the ranking performance on the test data.

    Args:
    - cfg (ExperimentConfig): ExperimentConfig object. See conf/kuairec.yml for details.
    """

    config = cfg.setting
    log_path = Path(f"./logs/{config.name}/")
    log_path.mkdir(exist_ok=True, parents=True)

    params_path = Path(f"./data/best_params/{config.name}")

    logger.info("start data loading...")
    dataloader = DataLoader(config, logger)
    plot_populality(dataloader.item_populality, config.name)

    logger.info("data loading is done.")

    model_params: dict = OmegaConf.to_container(config.model_params)

    if config.is_search_params:
        search_params(
            model_params=model_params,
            seed=config.seed,
            dataloader=dataloader,
            logger=logger,
            params_path=params_path,
            log_path=log_path,
        )

    logger.info("start experiment...")

    evaluator = TestEvaluator(
        interaction_df=dataloader.test_df,
        features=dataloader.test_evaluation_features,
        n_items=dataloader.n_items,
        used_metrics={QUALITATIVE_METRIC, QUANTITATIVE_METRIC},
        K=K,
    )

    metric_df = pd.DataFrame()
    for model_name in MODELS:
        for estimator in ESTIMATORS:
            base_name = f"{model_name}_{estimator}"

            train, val = dataloader.load(model_name=model_name, estimator=estimator)

            with open(params_path / f"{base_name}.json", "r") as f:
                search_results = json.load(f)

            if model_name == "FM":
                model = FM(
                    estimator=estimator,
                    n_epochs=search_results["n_epochs"],
                    n_factors=model_params["n_factors"],
                    n_features=train["features"].shape[1],
                    lr=model_params["lr"][model_name][estimator],
                    batch_size=model_params["batch_size"],
                    seed=config.seed,
                )

            elif model_name == "MF":
                model = MF(
                    estimator=estimator,
                    n_epochs=search_results["n_epochs"],
                    n_factors=model_params["n_factors"],
                    n_users=dataloader.n_users,
                    n_items=dataloader.n_items,
                    lr=model_params["lr"][model_name][estimator],
                    reg=model_params["reg"],
                    batch_size=model_params["batch_size"],
                    seed=config.seed,
                )

            _ = model.fit(train, val)

            test_pred_y = model.predict(X=evaluator.features[model_name])
            results = evaluator.evaluate(test_pred_y)
            for metric_name, values in results.items():
                metric_df[f"{base_name}_{metric_name}@K"] = values

            logger.info(f"{base_name} is done.")

    # random baseline
    model_name = "Random"
    np.random.seed(config.seed)
    results = evaluator.evaluate(
        y_scores=np.random.uniform(0, 1, size=len(dataloader.test_df))
    )
    for metric_name, values in results.items():
        metric_df[f"Random_{metric_name}@K"] = values

    logger.info(f"{model_name} is done.")

    metric_df.to_csv(log_path / "metric.csv", index=False)

    # visualize and save the results
    Visualizer(
        result_df=metric_df,
        K=K,
        quantitative_metrics=QUANTITATIVE_METRIC,
        qualitative_metrics=QUALITATIVE_METRIC,
        log_path=log_path / "test",
    )


if __name__ == "__main__":
    main()
