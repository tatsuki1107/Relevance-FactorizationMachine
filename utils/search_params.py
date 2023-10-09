from omegaconf import OmegaConf
from utils.dataloader.loader import DataLoader
from src.fm import FactorizationMachines as FM
from src.mf import LogisticMatrixFactorization as MF
from utils.evaluate import Evaluator
from conf.config import ModelConfig
from logging import Logger
from pathlib import Path
import numpy as np
from time import time
import json

VALUE_ERROR_MESSAGE = (
    "value_range must be tuple of int or float."
    + "param_name: {}, value_range: {}"
    + "You need to rewrite conf/config.yaml"
)

LOGGER_INFO_MESSAGE = (
    "model: {}, estimator: {}, epoch: {}, params: {}," + "{}@{}: {}"
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
    for param_name, value_range in model_config.items():
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
    n_epochs: int = 10,
    K: int = 3,
    used_metrics: str = "DCG",
) -> None:
    """ランダムサーチを実行する関数

    Args:
    - model_config (ModelConfig): モデルごとのパラメータ探索範囲の設定
    - seed (int): 乱数シード (read only)
    - dataloader (DataLoader): DataLoaderクラスのインスタンス
    - logger (Logger): Loggerクラスのインスタンス
    - n_epochs (int, optional): パラメータをサーチするエポック数. デフォルトは10.
    - K (int, optional):  最適化するランキング位置. デフォルトは3.
    - used_metrics (str, optional): 最適化する評価指標. デフォルトは"DCG".
    """

    model_config = OmegaConf.to_container(model_config, resolve=True)
    user2data_indices = dataloader.val_user2data_indices

    log_path = Path("./data/uniform_init_best_params")
    log_path.mkdir(exist_ok=True, parents=True)

    logger.info("start random search...")

    for model_name in ["FM", "MF"]:
        for estimator in ["Ideal", "IPS", "Naive"]:
            (
                train,
                val,
                _,
            ) = dataloader.load(model_name=model_name, estimator=estimator)

            evaluator = Evaluator(
                X=val[0],
                y_true=val[1],
                indices_per_user=user2data_indices,
                used_metrics=set([used_metrics]),
                K=[K],
                thetahold=0.75,
            )

            results = []
            for epoch in range(n_epochs):
                model_params = _get_params(
                    model_config=model_config[model_name], logger=logger
                )

                if estimator == "IPS":
                    # pscore clipping
                    train[2] = np.maximum(train[2], model_params["clipping"])
                    val[2] = np.maximum(val[2], model_params["clipping"])
                else:
                    model_params.pop("clipping")

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

                _, _ = model.fit(train, val)
                metrics = evaluator.evaluate(model, pscores=val[2])

                results.append((model_params, metrics[used_metrics][0]))

                logger.info(
                    LOGGER_INFO_MESSAGE.format(
                        model_name,
                        estimator,
                        epoch,
                        model_params,
                        used_metrics,
                        K,
                        metrics[used_metrics][0],
                    )
                )

            if estimator == "IPS":
                is_reverse = False
            else:
                is_reverse = True

            best_params = sorted(
                results, key=lambda x: x[1], reverse=is_reverse
            )[0]
            logger.info(f"best params: {best_params}")

            base_name = f"{model_name}_{estimator}"
            with open(log_path / f"{base_name}_best_param.json", "w") as f:
                json.dump(best_params[0], f)

    logger.info("random search is done.")
