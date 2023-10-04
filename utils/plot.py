import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import Set, List
from pathlib import Path
from dataclasses import dataclass, field


def plot_heatmap(matrix: np.ndarray) -> None:
    """評価値matrixをヒートマップで可視化
    args:
        matrix: 評価値行列
    """
    fig, ax = plt.subplots(figsize=(20, 5))

    my_cmap = plt.cm.get_cmap("Reds")
    heatmap = plt.pcolormesh(matrix.T, cmap=my_cmap)
    plt.colorbar(heatmap)
    ax.grid()
    plt.tight_layout()
    plt.show()


@dataclass
class Visualizer:
    metrics: Set[str]
    K: List[int]
    result_df: pd.DataFrame
    estimators: List[str] = field(
        default_factory=lambda: ["Ideal", "IPS", "Naive"]
    )

    def __post_init__(self):
        sns.set()
        self.log_path = Path("./logs/result/img")
        self.log_path.mkdir(exist_ok=True, parents=True)

        self._plot_metrics_per_model()
        self._plot_metric_vs_model()
        self._plot_metric_per_frequency()

    def _plot_metrics_per_model(self) -> None:
        for model_name in ["FM", "MF"]:
            plt.figure(figsize=(20, 6))
            for i, metric in enumerate(self.metrics):
                plt.subplot(1, len(self.metrics), i + 1)
                plt.title(f"{metric}@K", fontdict=dict(size=20))

                for estimator in self.estimators:
                    column = f"{model_name}_{estimator}_all_{metric}@K"
                    plt.scatter(self.K, self.result_df[column], marker="o")
                    plt.plot(self.K, self.result_df[column], label=estimator)

                column = f"Random_all_{metric}@K"
                plt.scatter(self.K, self.result_df[column], marker="o")
                plt.plot(self.K, self.result_df[column], label="Random")

                plt.xticks(self.K)
                plt.xlabel("varying K")
                plt.legend(loc="best", fontsize=20)

            plt.show()
            plt.savefig(self.log_path / f"{model_name}_metrics.png")

    def _plot_metric_vs_model(self, metric: str = "DCG"):
        plt.figure(figsize=(20, 6))

        for i, estimator in enumerate(self.estimators):
            plt.subplot(1, len(self.estimators), i + 1)
            plt.title(f"{estimator}: FM vs MF", fontdict=dict(size=20))

            for model_name in ["FM", "MF"]:
                column = f"{model_name}_{estimator}_all_{metric}@K"
                plt.scatter(self.K, self.result_df[column], marker="o")
                plt.plot(self.K, self.result_df[column], label=model_name)

            plt.xticks(self.K)
            plt.xlabel("varying K")
            plt.ylabel(f"{metric}@K")
            plt.legend(loc="best", fontsize=20)

        plt.show()
        plt.savefig(self.log_path / f"{metric}_vs_model.png")

    def _plot_metric_per_frequency(
        self, metric: str = "DCG", model_name: str = "FM"
    ):
        frequencies = ["all", "popular", "rare"]
        plt.figure(figsize=(20, 6))

        for i, freq in enumerate(frequencies):
            plt.subplot(1, len(frequencies), i + 1)
            plt.title(f"{freq} items", fontdict=dict(size=20))

            for estimator in self.estimators:
                column = f"{model_name}_{estimator}_{freq}_{metric}@K"
                plt.scatter(self.K, self.result_df[column], marker="o")
                plt.plot(self.K, self.result_df[column], label=estimator)

            column = f"Random_{freq}_{metric}@K"
            plt.scatter(self.K, self.result_df[column], marker="o")
            plt.plot(self.K, self.result_df[column], label="Random")

            plt.xticks(self.K)
            plt.xlabel("varying K")
            plt.ylabel(f"{metric} of {model_name}")
            plt.legend(loc="best", fontsize=20)

        plt.show()
        plt.savefig(self.log_path / f"{metric}_per_frequency.png")
