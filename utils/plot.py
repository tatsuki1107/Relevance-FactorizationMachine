# Standard library imports
from typing import List
from pathlib import PosixPath
from dataclasses import dataclass

# Third-party library imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


@dataclass
class Visualizer:
    """Visualizer for evaluating the performance of recommendation models

    Args:
    - K (List[int]): list of K
    - result_df (pd.DataFrame): result of evaluation
    - log_path (PosixPath): path to save the log
    - quantitative_metrics (str): quantitative metrics
    - qualitative_metrics (str): qualitative metrics
    """

    K: List[int]
    result_df: pd.DataFrame
    log_path: PosixPath
    quantitative_metrics: str
    qualitative_metrics: str

    def __post_init__(self):
        """initialize the Visualizer class"""

        sns.set()
        self.log_path.mkdir(exist_ok=True, parents=True)

        self.colors = {"FM": "#1f77b4", "MF": "#ff7f0e", "Random": "#2ca02c"}
        self.linestyles = {"IPS": "-", "Naive": "--", "Random": "-."}

        self._plot_vs_model()
        self._plot_mean_exposure()
        self._plot_metric_vs_estimator(self.quantitative_metrics)

    def _plot_vs_model(self, estimator="IPS") -> None:
        """plot metrics MF vs FM."""

        plt.figure(figsize=(20, 6))
        for i, metric_name in enumerate(
            [self.quantitative_metrics, self.qualitative_metrics]
        ):
            plt.subplot(1, 2, i + 1)
            for model_name in ["FM", "MF"]:
                col = f"{model_name}_{estimator}_{metric_name}@K"
                plt.title(f"{metric_name}@K", fontsize=25)
                plt.plot(
                    self.K,
                    self.result_df[col],
                    marker="o",
                    label=f"{model_name}_{estimator}",
                    color=self.colors[model_name],
                    linestyle=self.linestyles[estimator],
                )

            col = f"Random_{metric_name}@K"
            plt.plot(
                self.K,
                self.result_df[col],
                marker="o",
                label="Random",
                color=self.colors["Random"],
                linestyle=self.linestyles["Random"],
            )

            plt.xticks(self.K, fontsize=15)
            plt.xlabel("varying K", fontsize=20)
            plt.legend(loc="best", fontsize=20)

        plt.tight_layout()
        plt.show()
        plt.savefig(self.log_path / "metrics_vs_model.png")
        plt.close()

    def _plot_mean_exposure(self) -> None:
        """plot mean exposure."""

        plt.figure(figsize=(15, 6))
        for model_name in ["FM", "MF"]:
            for estimator in ["IPS", "Naive"]:
                col = f"{model_name}_{estimator}_ME@K"
                plt.title("Mean Exposure @ K", fontsize=25)
                plt.plot(
                    self.K,
                    self.result_df[col],
                    marker="o",
                    label=f"{model_name}_{estimator}",
                    color=self.colors[model_name],
                    linestyle=self.linestyles[estimator],
                )

        col = "Random_ME@K"
        plt.plot(
            self.K,
            self.result_df[col],
            marker="o",
            label="Random",
            color=self.colors["Random"],
            linestyle=self.linestyles["Random"],
        )

        plt.xticks(self.K, fontsize=15)
        plt.xlabel("varying K", fontsize=20)
        plt.legend(loc="best", fontsize=15)
        plt.show()
        plt.savefig(self.log_path / "mean_exposure.png")
        plt.close()

    def _plot_metric_vs_estimator(self, metric_name: str) -> None:
        """plot metric. IPS vs Naive.

        Args:
            metric_name (str): metric name
        """

        plt.figure(figsize=(20, 6))
        for i, model_name in enumerate(["FM", "MF"]):
            plt.subplot(1, 2, i + 1)
            for estimator in ["IPS", "Naive"]:
                col = f"{model_name}_{estimator}_{metric_name}@K"
                plt.title(f"{metric_name}@K Of {model_name}", fontsize=25)
                plt.plot(
                    self.K,
                    self.result_df[col],
                    marker="o",
                    label=f"{model_name}_{estimator}",
                    color=self.colors[model_name],
                    linestyle=self.linestyles[estimator],
                )

            col = f"Random_{metric_name}@K"
            plt.plot(
                self.K,
                self.result_df[col],
                marker="o",
                label="Random",
                color=self.colors["Random"],
                linestyle=self.linestyles["Random"],
            )

            plt.xticks(self.K, fontsize=15)
            plt.xlabel("varying K", fontsize=20)
            plt.legend(loc="best", fontsize=20)

        plt.tight_layout()
        plt.show()
        plt.savefig(self.log_path / "metric_vs_estimator.png")
        plt.close()


def plot_loss_curve(
    train_loss: list, val_loss: list, model_name: str, loss_img_path: PosixPath
) -> None:
    """Plot loss curve

    Args:
    - train_loss (list): train loss per epoch
    - val_loss (list): val loss per epoch
    - model_name (str): model name
    - loss_img_path (PosixPath): path to save the loss curve
    """

    loss_img_path.mkdir(exist_ok=True, parents=True)

    sns.set()
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label=f"train (last loss: {train_loss[-1]:.2f})")
    plt.plot(val_loss, label=f"val (last loss: {val_loss[-1]:.2f})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"Loss Curve of {model_name}", fontdict=dict(size=22))

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{loss_img_path}/{model_name}.png")
    plt.close()


def plot_val_metric_curve(
    metric_name: str, val_metric: list, model_name: str, metric_img_path: PosixPath
) -> None:
    """Plot validation metric (DCG) curve

    Args:
    - metric_name (str): metric name
    - val_metric (list): val metric per epoch
    - model_name (str): model name
    - metric_img_path (PosixPath): path to save the metric curve
    """

    metric_img_path.mkdir(exist_ok=True, parents=True)

    sns.set()
    plt.figure(figsize=(10, 6))
    plt.plot(val_metric, label=f"val (max value: {max(val_metric):.2f})")
    plt.xlabel("epoch")
    plt.ylabel(f"{metric_name}@K")
    plt.legend()
    plt.title(f"Val {metric_name}@K Curve of {model_name}", fontdict=dict(size=22))

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{metric_img_path}/{model_name}.png")
    plt.close()


def plot_populality(populalities: np.ndarray, data_name: str) -> None:
    """Plot item populality

    Args:
    - populalities (np.ndarray): item populality
    - data_name (str): data name. kuairec or coat.
    """

    sns.set()
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(populalities)[::-1])
    plt.xlabel("Sorted Num Of Video_id")
    plt.ylabel("Sum Of Label ")
    plt.title("Item Populality", fontdict=dict(size=22))

    plt.tight_layout()
    plt.show()
    plt.savefig(f"./data/{data_name}_item_populality.png")
    plt.close()
