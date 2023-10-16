import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from typing import Set, List
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Visualizer:
    """実験結果の描画をするためのクラス

    Args:
    - metrics (Set[str]): 使用する評価指標の集合
    - K (List[int]): ランク指標を算出するランキング位置
    - result_df (pd.DataFrame): 実験結果
    - estimators (List[str]): 評価する推定量
    """

    metrics: Set[str]
    K: List[int]
    result_df: pd.DataFrame
    estimators: List[str] = field(
        default_factory=lambda: ["Ideal", "IPS", "Naive"]
    )

    def __post_init__(self):
        """描画の実行"""

        sns.set()
        self.log_path = Path("./logs/result/img")
        self.log_path.mkdir(exist_ok=True, parents=True)

        self._plot_metrics_per_model()
        self._plot_metric_vs_model()
        self._plot_metric_per_frequency()
        self._plot_snips_estimated_values()

    def _plot_metrics_per_model(self) -> None:
        """モデル(FM, MF)ごとのランク指標を描画して保存する"""

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
                plt.plot(
                    self.K,
                    self.result_df[column],
                    label="Random",
                    linestyle="--",
                )

                plt.xticks(self.K)
                plt.xlabel("varying K")
                plt.legend(loc="best", fontsize=20)

            plt.show()
            plt.savefig(self.log_path / f"{model_name}_metrics.png")

    def _plot_metric_vs_model(self, metric: str = "DCG"):
        """MFとFMの性能を比べるための描画

        Args:
            metric (str, optional): 比較するランク指標。デフォルトでは、DCG@K
        """

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
        self, model_name: str = "FM", frequency: str = "rare"
    ):
        """露出頻度ごとのモデルのランク性能を描画

        Args:
            metric (str, optional): 評価する指標。デフォルトでは、DCG@K
            model_name (str, optional): 評価するモデル。デフォルトでは、FM
        """
        plt.figure(figsize=(20, 6))

        for i, metric in enumerate(self.metrics):
            plt.subplot(1, len(self.metrics), i + 1)
            plt.title(f"{frequency}: {metric}@K", fontdict=dict(size=20))

            for estimator in self.estimators:
                column = f"{model_name}_{estimator}_{frequency}_{metric}@K"
                plt.scatter(self.K, self.result_df[column], marker="o")
                plt.plot(self.K, self.result_df[column], label=estimator)

            column = f"Random_{frequency}_{metric}@K"
            plt.scatter(self.K, self.result_df[column], marker="o")
            plt.plot(
                self.K, self.result_df[column], label="Random", linestyle="--"
            )

            plt.xticks(self.K)
            plt.xlabel("varying K")
            plt.legend(loc="best", fontsize=20)

        plt.show()
        plt.savefig(self.log_path / "metrics_per_frequency.png")

    def _plot_snips_estimated_values(self, metric_name: str = "DCG") -> None:
        metric_path = Path("./data/best_params")
        with open(metric_path / "Random_baseline.json", "r") as f:
            random_metric: dict = json.load(f)

        with open(metric_path / "FM_IPS_best_param.json", "r") as f:
            fm_metric: dict = json.load(f)
        fm_metric.pop("params")

        with open(metric_path / "MF_IPS_best_param.json", "r") as f:
            mf_metric: dict = json.load(f)
        mf_metric.pop("params")

        width = 0.5
        plt.figure(figsize=(10, 6))
        plt.bar(0, fm_metric[metric_name], width, label="FM")
        plt.bar(1, mf_metric[metric_name], width, label="MF")
        plt.axhline(
            random_metric["SNIPS_DCG"],
            linestyle="--",
            color="red",
            label="Random",
        )
        plt.xticks([0, 1], ["FM", "MF"])
        plt.xlabel("varying model")
        plt.ylabel(f"SNIPS {metric_name} value")
        plt.legend()
        plt.title(
            "Self Normalized Inversed Propensity Estimator of "
            + f"{metric_name} per Model"
        )
        plt.show()
        plt.savefig(self.log_path / "snips_estimator.png")
