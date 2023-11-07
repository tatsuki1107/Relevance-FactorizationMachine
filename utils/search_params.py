# Standard library imports
from logging import Logger
from pathlib import Path
from time import time
import json
from collections import defaultdict

# Third-party library imports
from omegaconf import OmegaConf
import numpy as np

# Internal modules imports
from utils.dataloader.loader import DataLoader
from utils.evaluate import Evaluator
from src.fm import FactorizationMachines as FM
from src.mf import LogisticMatrixFactorization as MF
from conf.config import ModelConfig


VALUE_ERROR_MESSAGE = (
    "value_range must be tuple of int or float."
    + "param_name: {}, value_range: {}"
    + "You need to rewrite conf/config.yaml"
)

MODEL_PARAMS_MESSAGE = (
    "model: {}, estimator: {}, trial: {}, params: {}, " + "{}@{}: {}"
)

MODEL_DISTRIBUTION_MESSAGE = (
    "log loss: {}, " + "prediction min: {}, max: {}, " + "mean: {}, std: {}"
)


def _get_params(model_config: dict, logger: Logger) -> dict:
    """パラメータ探索範囲からランダムにパラメータをサンプリングする関数

    Args:
    - model_config (dict): モデルごとのパラメータ探索範囲の設定
    - logger (Logger): Loggerクラスのインスタンス

    Raises:
        ValueError: パラメータがintかfloatの範囲でない場合

    Returns:
        dict: パラメータの辞書
    """

    dynamic_seed = int(time())
    np.random.seed(dynamic_seed)
    params = {}
    for param_name, values in model_config.items():
        value_range = list(values.values())
        if all(isinstance(value, int) for value in value_range):
            params[param_name] = np.random.randint(
                value_range[0], value_range[1]
            )
        elif all(isinstance(value, float) for value in value_range):
            params[param_name] = np.random.uniform(
                value_range[0], value_range[1]
            )
        else:
            logger.error(VALUE_ERROR_MESSAGE.format(param_name, value_range))
            raise ValueError(
                VALUE_ERROR_MESSAGE.format(param_name, value_range)
            )
    return params


def random_search(
    model_config: ModelConfig,
    seed: int,
    dataloader: DataLoader,
    logger: Logger,
    n_trials: int = 100,
    K: int = 3,
    used_metrics: str = "DCG",
) -> None:
    """ランダムサーチを実行する関数

    Args:
    - model_config (ModelConfig): モデルごとのパラメータ探索範囲の設定
    - seed (int): 乱数シード (read only)
    - dataloader (DataLoader): DataLoaderクラスのインスタンス
    - logger (Logger): Loggerクラスのインスタンス
    - n_trials (int, optional): パラメータをサーチするエポック数. デフォルトは100.
    - K (int, optional):  最適化するランキング位置. デフォルトは3.
    - used_metrics (str, optional): 最適化する評価指標. デフォルトは"DCG".
    """

    model_config = OmegaConf.to_container(model_config)

    log_path = Path("./data/best_params")
    log_path.mkdir(exist_ok=True, parents=True)

    # random baseline
    model_name = "Random"
    random_val_data = dataloader.val_data_for_random_policy
    user2data_indices = dataloader.val_user2data_indices
    dumped_metric = defaultdict(dict)
    for estimator, val_data in random_val_data.items():
        evaluator = Evaluator(
            _seed=seed,
            X=None,
            y_true=val_data["y_true"],
            indices_per_user=user2data_indices,
            used_metrics=set([used_metrics]),
            K=[K],
            thetahold=None
        )
        metrics = evaluator.evaluate(
            model=model_name, pscores=val_data["pscore"]
        )
        dumped_metric[estimator][model_name] = metrics[used_metrics][0]
        logger.info(
            f"{used_metrics} of Random_{estimator}: {metrics[used_metrics][0]}"
        )

    logger.info("start random search...")

    for model_name in ["FM", "MF"]:
        for estimator in ["Ideal", "IPS", "Naive"]:
            (
                train,
                val,
                _,
            ) = dataloader.load(model_name=model_name, estimator=estimator)

            evaluator = Evaluator(
                _seed=seed,
                X=val[0],
                y_true=val[1],
                indices_per_user=user2data_indices,
                used_metrics=set([used_metrics]),
                K=[K],
                thetahold=None
            )
            # search_results = [(trial, params, metric),(...),(...)]
            search_results = []
            for trial in range(n_trials):
                model_params = _get_params(
                    model_config=model_config[model_name], logger=logger
                )

                if model_name == "FM":
                    model = FM(
                        n_epochs=model_params["n_epochs"],
                        n_factors=model_params["n_factors"],
                        n_features=train[0].shape[1],
                        lr=model_params["lr"],
                        batch_size=model_params["batch_size"],
                        seed=seed,
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
                        seed=seed,
                    )

                _, val_loss = model.fit(train, val)
                metrics = evaluator.evaluate(model, pscores=val[2])
                search_results.append((
                    trial,
                    model_params,
                    metrics[used_metrics][0]))

                logger.info(
                    MODEL_PARAMS_MESSAGE.format(
                        model_name,
                        estimator,
                        trial,
                        model_params,
                        used_metrics,
                        K,
                        metrics[used_metrics][0],
                    )
                )

                pred_scores = model.predict(val[0])
                logger.info(
                    MODEL_DISTRIBUTION_MESSAGE.format(
                        val_loss[-1],
                        pred_scores.min(),
                        pred_scores.max(),
                        pred_scores.mean(),
                        pred_scores.std(),
                    )
                )

            best_result = sorted(
                search_results, key=lambda x: x[2], reverse=True
            )[0]
            best_result = dict(zip(
                ("trial", "params", used_metrics), best_result)
            )
            logger.info(f"best result: {best_result}")

            base_name = f"{model_name}_{estimator}"
            with open(log_path / f"{base_name}_best_param.json", "w") as f:
                json.dump(best_result["params"], f)

            dumped_metric[estimator][model_name] = best_result[used_metrics]
            with open(log_path / f"best_{used_metrics}.json", "w") as f:
                json.dump(dumped_metric, f)

    logger.info("random search is done.")
