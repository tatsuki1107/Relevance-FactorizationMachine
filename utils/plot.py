# Standard library imports
import json
from typing import List
from pathlib import Path
from dataclasses import dataclass, field

# Third-party library imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


@dataclass
class Visualizer:
    """実験結果の描画をするためのクラス

    Args:
    - metrics (Set[str]): 使用する評価指標の集合
    - K (List[int]): ランク指標を算出するランキング位置
    - result_df (pd.DataFrame): 実験結果
    - estimators (List[str]): 評価する推定量
    """

    K: List[int]
    result_df: pd.DataFrame
    estimators: List[str] = field(
        default_factory=lambda: ["Ideal", "IPS", "Naive"]
    )
    metrics: List[str] = field(
        default_factory=lambda: ["DCG", "Recall", "MAP"]
    )

    def __post_init__(self):
        """描画の実行"""

        sns.set()
        self.log_path = Path("./logs/result/img")
        self.log_path.mkdir(exist_ok=True, parents=True)

        self._plot_metrics_per_model()
        self._plot_metric_vs_model()
        self._plot_metric_per_frequency()
        self._plot_ips_estimated_values()
        self._plot_mean_exposure()

    def _plot_metrics_per_model(self) -> None:
        """モデル(FM, MF)ごとのランク指標を描画して保存する"""

        for model_name in ["FM", "MF"]:
            plt.figure(figsize=(20, 6))
            for i, metric in enumerate(self.metrics):
                plt.subplot(1, len(self.metrics), i + 1)
                plt.title(f"{metric}@K", fontdict=dict(size=25))

                for estimator in self.estimators:
                    column = f"{model_name}_{estimator}_all_{metric}@K"
                    self._plot_and_scatter(
                        df_column=column,
                        label=f"{model_name}_{estimator}",
                    )

                column = f"Random_all_{metric}@K"
                self._plot_and_scatter(
                    df_column=column,
                    label="Random",
                    linestyle="--",
                )

                plt.xticks(self.K)
                plt.xlabel("varying K")
                plt.legend(loc="best", fontsize=20)

            plt.tight_layout()
            plt.show()
            plt.savefig(self.log_path / f"{model_name}_metrics.png")
            plt.close()

    def _plot_metric_vs_model(self, metric: str = "DCG"):
        """MFとFMの性能を比べるための描画

        Args:
            metric (str, optional): 比較するランク指標。デフォルトでは、DCG@K
        """

        plt.figure(figsize=(20, 6))

        for i, estimator in enumerate(self.estimators):
            plt.subplot(1, len(self.estimators), i + 1)
            plt.title(
                f"{metric}@K: FM_{estimator} vs MF_{estimator}",
                fontdict=dict(size=25),
            )

            for model_name in ["FM", "MF"]:
                column = f"{model_name}_{estimator}_all_{metric}@K"
                self._plot_and_scatter(
                    df_column=column,
                    label=f"{model_name}_{estimator}",
                )

            column = f"Random_all_{metric}@K"
            self._plot_and_scatter(
                df_column=column,
                label="Random",
                linestyle="--",
            )

            plt.xticks(self.K)
            plt.xlabel("varying K")
            plt.legend(loc="best", fontsize=20)

        plt.tight_layout()
        plt.show()
        plt.savefig(self.log_path / f"{metric}_vs_model.png")
        plt.close()

    def _plot_mean_exposure(self, metric: str = "ME") -> None:
        """カスタム評価指標Mean Exposure @ Kの描画

        Args:
            metric (str, optional): 比較するランク指標。
        """

        plt.figure(figsize=(20, 6))
        for i, model_name in enumerate(["FM", "MF"]):
            plt.subplot(1, 3, i + 1)
            plt.title(
                f"Mean Exposure@K of {model_name}", fontdict=dict(size=25)
            )

            for estimator in self.estimators:
                column = f"{model_name}_{estimator}_all_{metric}@K"
                self._plot_and_scatter(
                    df_column=column,
                    label=f"{model_name}_{estimator}",
                )

            column = f"Random_all_{metric}@K"
            self._plot_and_scatter(
                df_column=column,
                label="Random",
                linestyle="--",
            )

            plt.xticks(self.K)
            plt.xlabel("varying K")
            plt.legend(loc="best", fontsize=15)

        plt.subplot(1, 3, 3)
        plt.title("Mean Exposure@K: FM_Naive vs MF_Naive", fontsize=20)
        for model_name in ["FM", "MF"]:
            column = f"{model_name}_Naive_all_{metric}@K"
            self._plot_and_scatter(
                df_column=column,
                label=f"{model_name}_Naive",
            )

        plt.xticks(self.K)
        plt.xlabel("varying K")
        plt.legend(loc="best", fontsize=20)

        plt.tight_layout()
        plt.show()
        plt.savefig(self.log_path / "mean_exposure.png")
        plt.close()

    def _plot_metric_per_frequency(
        self, model_name: str = "FM", frequency: str = "rare"
    ):
        """露出頻度ごとのモデルのランク性能を描画

        Args:
            model_name (str): 比較するモデル。デフォルトでは、FM
            frequency (str): 比較する露出頻度。デフォルトでは、rare
        """

        plt.figure(figsize=(20, 6))

        for i, metric in enumerate(self.metrics):
            plt.subplot(1, len(self.metrics), i + 1)
            plt.title(
                f"{metric}@K: {frequency} items only", fontdict=dict(size=25)
            )

            for estimator in self.estimators:
                column = f"{model_name}_{estimator}_{frequency}_{metric}@K"
                self._plot_and_scatter(
                    df_column=column,
                    label=f"{model_name}_{estimator}",
                )

            column = f"Random_{frequency}_{metric}@K"
            self._plot_and_scatter(
                df_column=column,
                label="Random",
                linestyle="--",
            )

            plt.xticks(self.K)
            plt.xlabel("varying K")
            plt.legend(loc="best", fontsize=20)

        plt.tight_layout()
        plt.show()
        plt.savefig(self.log_path / "metrics_per_frequency.png")
        plt.close()

    def _plot_ips_estimated_values(self, metric_name: str = "DCG") -> None:
        """MFとFMの検証データにおけるIPS推定値を描画

        Args:
            metric_name (str): 描画するランク指標。デフォルトでは、DCG
        """
        metric_path = Path("./data/best_params")
        with open(metric_path / f"best_{metric_name}.json", "r") as f:
            val_metric: dict = json.load(f)

        ips_val_metrics = val_metric["IPS"]

        width = 0.5
        plt.figure(figsize=(10, 6))
        plt.bar(0, ips_val_metrics["FM"], width, label="FM")
        plt.bar(1, ips_val_metrics["MF"], width, label="MF")
        plt.axhline(
            ips_val_metrics["Random"],
            linestyle="--",
            color="r",
            label="Random",
        )
        plt.xticks([0, 1], ["FM", "MF"])
        plt.xlabel("varying model")
        plt.ylabel(f"IPS of {metric_name} value")
        plt.legend()
        plt.title(
            "IPS Estimator of " + f"{metric_name} per Model",
            fontdict=dict(size=22),
        )

        plt.tight_layout()
        plt.show()
        plt.savefig(self.log_path / f"IPS_estimator_of_{metric_name}.png")
        plt.close()

    def _plot_and_scatter(
        self,
        df_column: str,
        label: str,
        linestyle: str = "-",
    ) -> None:
        """折れ線グラフと散布図を描画する

        Args:
            df_column (str): 描画するデータフレームの列名
            label (str): 凡例のラベル
            linestyle (str, optional): 折れ線グラフのスタイル。デフォルトでは、実線
        """

        plt.scatter(
            self.K,
            self.result_df[df_column],
            marker="o",
        )
        plt.plot(
            self.K,
            self.result_df[df_column],
            label=label,
            linestyle=linestyle,
        )


def plot_loss_curve(train_loss: list, val_loss: list, model_name: str) -> None:
    """学習曲線を描画する関数

    Args:
        train_loss (list): ミニバッチデータのエポックごとの損失値の平均と標準偏差
        val_loss (list): 検証データのエポックごとの損失値
        model_name (str): モデルの名前
    """
    sns.set()
    plt.figure(figsize=(10, 6))
    mean_train_loss, std_train_loss = map(np.array, zip(*train_loss))
    epochs = range(len(mean_train_loss))
    plt.plot(
        mean_train_loss, label=f"train (last loss: {mean_train_loss[-1]:.2f})"
    )
    plt.fill_between(
        epochs,
        mean_train_loss - std_train_loss,
        mean_train_loss + std_train_loss,
        color="blue",
        alpha=0.1,
    )

    plt.plot(val_loss, label=f"val (last loss: {val_loss[-1]:.2f})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"Loss Curve of {model_name}", fontdict=dict(size=22))

    plt.tight_layout()
    plt.show()
    plt.savefig(f"./data/loss_curve/{model_name}.png")
    plt.close()


def plot_exposure(exposure_probabilities: np.ndarray) -> None:
    """露出度の確率を描画する関数

    Args:
        exposure_probabilities (np.ndarray): 露出度の確率
    """
    sns.set()
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(exposure_probabilities)[::-1])
    plt.xlabel("sorted num of video_id")
    plt.ylabel("exposure probability")
    plt.title("Exposure Probability Distribution", fontdict=dict(size=22))

    plt.tight_layout()
    plt.show()
    plt.savefig("./data/exposure_probability.png")
    plt.close()
