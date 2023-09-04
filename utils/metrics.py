import numpy as np
from typing import Tuple


def calc_precision_at_k(
    y_true: np.ndarray, y_scores: np.ndarray, k: int
) -> float:
    y_true_sorted_by_scores = y_true[y_scores.argsort()[::-1]][:k]
    return np.mean(y_true_sorted_by_scores)


def calc_recall_at_k(
    y_true: np.ndarray, y_scores: np.ndarray, k: int
) -> float:
    y_true_sorted_by_scores = y_true[y_scores.argsort()[::-1]]

    recall = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        recall = np.sum(y_true_sorted_by_scores[:k]) / np.sum(
            y_true_sorted_by_scores
        )

    return recall


def calc_dcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    y_true_sorted_by_scores = y_true[y_scores.argsort()[::-1]]

    dcg_score = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        dcg_score += y_true_sorted_by_scores[0]

        molecules = y_true_sorted_by_scores[1:k]
        indices = np.arange(1, molecules.shape[0] + 1)
        denominator = np.log2(indices + 1)
        dcg_score += np.sum(molecules / denominator)

    return dcg_score


def calc_roc_auc(
    y_true: np.ndarray, y_scores: np.ndarray, thetahold: np.ndarray
) -> Tuple[list, list, float]:
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
