# Standard library imports
from typing import Tuple, Callable, Dict, Union

# Third-party library imports
import numpy as np


def calc_average_precision_at_k(
    y_true_sorted_by_scores: np.ndarray,
    k: int,
) -> float:
    """Average Precision@kを計算する

    Args:
    - y_true_sorted_by_scores (np.ndarray): ソート済み正解ラベルの配列
    - k (int): 上位k件のみを対象とする

    Returns:
    - (float): Average Precision@kの値
    """

    average_precision = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        for i in range(min(k, len(y_true_sorted_by_scores))):
            if y_true_sorted_by_scores[i] == 1:
                average_precision += np.sum(
                    y_true_sorted_by_scores[: i + 1]
                ) / (i + 1)

    return average_precision


def calc_recall_at_k(
    y_true_sorted_by_scores: np.ndarray,
    k: int,
) -> float:
    """Recall@kを計算する

    Args:
    - y_true_sorted_by_scores (np.ndarray): ソート済み正解ラベルの配列
    - k (int): 上位k件のみを対象とする

    Returns:
    - (float): Recall@kの値
    """

    recall = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        recall = np.sum(y_true_sorted_by_scores[:k]) / np.sum(
            y_true_sorted_by_scores
        )

    return recall


def calc_ips_of_dcg_at_k(
    y_true_sorted_by_scores: np.ndarray,
    k: int,
    pscores_sorted_by_scores: np.ndarray,
) -> float:
    """DCG@kのIPS推定値を計算する

    Args:
    - y_true_sorted_by_scores (np.ndarray): ソート済み正解ラベルの配列
    - k (int): 上位k件のみを対象とする
    - pscores_sorted_by_scores (np.ndarray): 傾向スコアの配列

    Returns:
    - float: DCG@kのIPS推定値。pscores_sorted_by_scoresがすべて1ならば、
            通常のDCG@kを返す。
    """

    dcg_score = 0.0
    if np.sum(y_true_sorted_by_scores) == 0:
        return np.nan
    else:
        dcg_score += y_true_sorted_by_scores[0] / pscores_sorted_by_scores[0]

        molecules = y_true_sorted_by_scores[1:k]
        indices = np.arange(1, molecules.shape[0] + 1)
        denominator = pscores_sorted_by_scores[1:k] * np.log2(indices + 1)
        dcg_score += np.sum(molecules / denominator)

    return dcg_score


def calc_dcg_at_k(
    y_true_sorted_by_scores: np.ndarray,
    k: int,
) -> float:
    """DCG@kのIPS推定値を計算する

    Args:
    - y_true_sorted_by_scores (np.ndarray): ソート済み正解ラベルの配列
    - k (int): 上位k件のみを対象とする

    Returns:
    - float: DCG@kの値
    """

    dcg_score = 0.0
    if np.sum(y_true_sorted_by_scores) == 0:
        return np.nan
    else:
        dcg_score += y_true_sorted_by_scores[0]
        molecules = y_true_sorted_by_scores[1:k]
        indices = np.arange(1, molecules.shape[0] + 1)
        denominator = np.log2(indices + 1)
        dcg_score += np.sum(molecules / denominator)

    return dcg_score


def return_exposure_at_k(
    pscores_sorted_by_scores: np.ndarray,
    k: int,
) -> Union[int, None]:
    """推薦位置kのモデルが出力したアイテムの露出確率を返す

    Args:
        pscores_sorted_by_scores (np.ndarray): ソート済み傾向スコアの配列
        k (int): 推薦位置

    Returns:
        Union[int, None]: 露出確率。ユーザの正解データがk件未満の場合は、np.nanを返す。
    """

    if len(pscores_sorted_by_scores) >= k:
        return pscores_sorted_by_scores[k - 1]
    else:
        return np.nan


def calc_roc_auc(
    y_true: np.ndarray, y_scores: np.ndarray, interval: float = 0.0001
) -> Tuple[list, list, float]:
    thetahold = np.arange(0, 1 + interval, interval)[::-1]
    tpr, fpr = [], []
    for theta in thetahold:
        y_pred = (y_scores >= theta).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        tpr.append(TPR)
        fpr.append(FPR)

    roc = 0.0
    for i in range(len(thetahold) - 1):
        roc += (tpr[i + 1] + tpr[i]) * (fpr[i + 1] - fpr[i]) / 2

    return tpr, fpr, roc


metric_candidates: Dict[str, Callable] = {
    "Recall": calc_recall_at_k,
    "MAP": calc_average_precision_at_k,
    "DCG": calc_dcg_at_k,
    "ME": return_exposure_at_k,
}
