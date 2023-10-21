# Standard library imports
from typing import Tuple, Callable, Dict

# Third-party library imports
import numpy as np

NOT_IMPLEMENTED_ERROR_MESSAGE = "Re-implement if for SNIPS estimator"


def calc_precision_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
    pscores: np.ndarray,
) -> float:
    """Precision@kを計算する

    Args:
    - y_true (np.ndarray): 正解ラベルの配列
    - y_scores (np.ndarray): 予測確率の配列
    - k (int): 上位k件のみを対象とする
    - pscores (np.ndarray): 傾向スコアの配列

    Raises:
    - NotImplementedError: pscoresが全て1の場合のみ実装済み

    Returns:
    - (float): Precision@kの値
    """
    y_true_sorted_by_scores = y_true[y_scores.argsort()[::-1]][:k]

    if np.all(pscores == 1):
        return np.mean(y_true_sorted_by_scores)
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)


def calc_average_precision_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
    pscores: np.ndarray,
) -> float:
    """Average Precision@kを計算する

    Args:
    - y_true (np.ndarray): 正解ラベルの配列
    - y_scores (np.ndarray): 予測確率の配列
    - k (int): 上位k件のみを対象とする
    - pscores (np.ndarray): 傾向スコアの配列

    Raises:
    - NotImplementedError: pscoresが全て1の場合のみ実装済み

    Returns:
    - (float): Average Precision@kの値
    """

    y_true_sorted_by_scores = y_true[y_scores.argsort()[::-1]]

    average_precision = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        for i in range(min(k, len(y_true_sorted_by_scores))):
            if y_true_sorted_by_scores[i] == 1:
                average_precision += np.sum(
                    y_true_sorted_by_scores[: i + 1]
                ) / (i + 1)

    if np.all(pscores == 1):
        return average_precision

    raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)


def calc_recall_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
    pscores: np.ndarray,
) -> float:
    """Recall@kを計算する

    Args:
    - y_true (np.ndarray): 正解ラベルの配列
    - y_scores (np.ndarray): 予測確率の配列
    - k (int): 上位k件のみを対象とする
    - pscores (np.ndarray): 傾向スコアの配列

    Raises:
    - NotImplementedError: pscoresが全て1の場合のみ実装済み

    Returns:
    - (float): Recall@kの値
    """
    y_true_sorted_by_scores = y_true[y_scores.argsort()[::-1]]

    recall = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        recall = np.sum(y_true_sorted_by_scores[:k]) / np.sum(
            y_true_sorted_by_scores
        )

    if np.all(pscores == 1):
        return recall

    raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)


def calc_dcg_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
    pscores: np.ndarray,
) -> float:
    """DCG@kを計算する

    Args:
    - y_true (np.ndarray): 正解ラベルの配列
    - y_scores (np.ndarray): 予測確率の配列
    - k (int): 上位k件のみを対象とする
    - pscores (np.ndarray): 傾向スコアの配列

    Returns:
    - float: DCG@kの値。pscoresがすべて1ではない場合は、DCG@KのSNIPS推定量を返す。
    すなわち、バイアスのかかったvalデータからIPS推定量を評価する。
    """

    y_true_sorted_by_scores = y_true[y_scores.argsort()[::-1]]

    pscores_sorted_by_scores = pscores[y_scores.argsort()[::-1]]

    dcg_score = 0.0
    if np.sum(y_true_sorted_by_scores) == 0:
        return np.nan
    else:
        dcg_score += y_true_sorted_by_scores[0] / pscores_sorted_by_scores[0]

        molecules = y_true_sorted_by_scores[1:k]
        indices = np.arange(1, molecules.shape[0] + 1)
        denominator = pscores_sorted_by_scores[1:k] * np.log2(indices + 1)
        dcg_score += np.sum(molecules / denominator)

    if np.all(pscores == 1):
        return dcg_score

    return dcg_score / np.sum(
        1.0 / pscores_sorted_by_scores[y_true_sorted_by_scores == 1]
    )


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
    "Precision": calc_precision_at_k,
    "MAP": calc_average_precision_at_k,
    "DCG": calc_dcg_at_k,
}
