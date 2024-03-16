# Standard library imports
from logging import Logger
from pathlib import PosixPath
import json

# Third-party library imports
import numpy as np

# Internal modules imports
from utils.dataloader.kuairec.loader import DataLoader
from utils.evaluate import ValEvaluator
from utils.plot import plot_loss_curve, plot_val_metric_curve
from src.fm import FactorizationMachines as FM
from src.mf import LogisticMatrixFactorization as MF


VALUE_ERROR_MESSAGE = (
    "value_range must be tuple of int or float."
    + "param_name: {}, value_range: {}"
    + "You need to rewrite conf/config.yaml"
)

MODEL_PARAMS_MESSAGE = "model: {}, estimator: {}, best_epoch: {}, " + "val {}@{}: {}"

MODEL_DISTRIBUTION_MESSAGE = (
    "log loss: {}, " + "prediction min: {}, max: {}, " + "mean: {}, std: {}"
)

ESTIMATOR = ["IPS", "Naive"]
MODEL = ["FM", "MF"]


def search_params(
    model_params: dict,
    seed: int,
    dataloader: DataLoader,
    logger: Logger,
    params_path: PosixPath,
    log_path: PosixPath,
    k: int = 5,
    max_epoch: int = 500,
    used_metric: str = "DCG",
) -> None:
    """search best hyper parameter (n_epoch only)

    Args:
        model_params (dict): decided hyper parameters
        seed (int): random seed
        dataloader (DataLoader): data loader instance
        logger (Logger): logger instance
        params_path (PosixPath): path to save best hyper parameter
        log_path (PosixPath): path to save loss curve and metric curve
        k (int, optional): recommend position. Defaults to 5.
        max_epoch (int, optional): max of num epoch. Defaults to 500.
        used_metric (str, optional): used metric. Defaults to "DCG".
    """

    # init evaluator
    evaluator = ValEvaluator(
        interaction_df=dataloader.val_df,
        features=dataloader.val_evaluation_features,
        k=k,
        metric_name=used_metric,
    )

    # random baseline
    model_name = "Random"
    np.random.seed(seed)
    y_scores = np.random.uniform(0, 1, dataloader.val_df.shape[0])

    params_path.mkdir(exist_ok=True, parents=True)

    for estimator in ESTIMATOR:
        metric_value = evaluator.evaluate(y_scores=y_scores, estimator=estimator)
        logger.info(f"{used_metric} of Random_{estimator}: {metric_value}")

    logger.info("start searching param...")

    for model_name in MODEL:
        for estimator in ESTIMATOR:
            train, val = dataloader.load(model_name=model_name, estimator=estimator)

            if model_name == "FM":
                model = FM(
                    estimator=estimator,
                    n_epochs=max_epoch,
                    n_factors=model_params["n_factors"],
                    n_features=train["features"].shape[1],
                    lr=model_params["lr"][model_name][estimator],
                    batch_size=model_params["batch_size"],
                    seed=seed,
                    evaluator=evaluator,
                )
            elif model_name == "MF":
                model = MF(
                    estimator=estimator,
                    n_epochs=max_epoch,
                    n_factors=model_params["n_factors"],
                    n_users=dataloader.n_users,
                    n_items=dataloader.n_items,
                    lr=model_params["lr"][model_name][estimator],
                    reg=model_params["reg"],
                    batch_size=model_params["batch_size"],
                    seed=seed,
                    evaluator=evaluator,
                )

            train_loss, val_loss = model.fit(train, val)
            plot_loss_curve(
                train_loss=train_loss,
                val_loss=val_loss,
                model_name=f"{model_name}_{estimator}",
                loss_img_path=log_path / "val" / "loss_curve",
            )
            plot_val_metric_curve(
                metric_name=used_metric,
                val_metric=model.val_metrics,
                model_name=f"{model_name}_{estimator}",
                metric_img_path=log_path / "val" / f"{used_metric}_curve",
            )

            best_epoch = int(np.argmax(model.val_metrics))
            metric_value = model.val_metrics[best_epoch]

            logger.info(
                MODEL_PARAMS_MESSAGE.format(
                    model_name,
                    estimator,
                    best_epoch,
                    used_metric,
                    k,
                    metric_value,
                )
            )

            logger.info(
                MODEL_DISTRIBUTION_MESSAGE.format(
                    val_loss[-1],
                    y_scores.min(),
                    y_scores.max(),
                    y_scores.mean(),
                    y_scores.std(),
                )
            )

            base_name = f"{model_name}_{estimator}"
            param_result = {
                "n_epochs": best_epoch,
                f"val_{used_metric}": metric_value,
            }
            with open(params_path / f"{base_name}.json", "w") as f:
                json.dump(param_result, f)

    logger.info("searching param is done.")
