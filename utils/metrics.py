# Standard library imports
from typing import Callable, Dict, Union, List
from collections import Counter

# Third-party library imports
import numpy as np


def calc_average_precision_at_k(
    y_true_sorted_by_scores: np.ndarray,
    k: int,
) -> float:
    """calculate average precision at k

    Args:
    - y_true_sorted_by_scores (np.ndarray): sorted true labels
    - k (int): recommend position

    Returns:
        float: average precision at k
    """

    average_precision = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        for i in range(min(k, len(y_true_sorted_by_scores))):
            if y_true_sorted_by_scores[i] >= 1:
                average_precision += np.sum(y_true_sorted_by_scores[: i + 1]) / (i + 1)

    return average_precision


def calc_recall_at_k(
    y_true_sorted_by_scores: np.ndarray,
    k: int,
) -> float:
    """calculate recall at k

    Args:
    - y_true_sorted_by_scores (np.ndarray): sorted true labels
    - k (int): recommend position

    Returns:
        float: recall at k
    """

    recall = 0.0
    if not np.sum(y_true_sorted_by_scores) == 0:
        recall = np.sum(y_true_sorted_by_scores[:k]) / np.sum(y_true_sorted_by_scores)

    return recall


def calc_ips_of_dcg_at_k(
    y_true_sorted_by_scores: np.ndarray,
    k: int,
    pscores_sorted_by_scores: np.ndarray,
) -> float:
    """calculate IPS estimation of DCG@k

    Args:
    - y_true_sorted_by_scores (np.ndarray): sorted true labels
    - k (int): recommend position
    - pscores_sorted_by_scores (np.ndarray): sorted propensity scores

    Returns:
        float: IPS estimation of DCG@k
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
    """calculate DCG at k

    Args:
    - y_true_sorted_by_scores (np.ndarray): sorted true labels
    - k (int): recommend position

    Returns:
        float: DCG at k
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
    """return exposure at k. return exposure probability of ranked kth item.

    Args:
    - pscores_sorted_by_scores (np.ndarray): sorted propensity scores
    - k (int): recommend position

    Returns:
        Union[int, None]: exposure at k
    """

    if len(pscores_sorted_by_scores) >= k:
        return pscores_sorted_by_scores[k - 1]
    else:
        return np.nan


def calc_gini_at_k(rec_items: List[int], n_items: int) -> float:
    """calculate Gini coefficient at k

    Args:
    - rec_items (List[int]): recommended items at k
    - n_items (int): number of items

    Returns:
        float: Gini coefficient at k
    """

    item_counter = Counter(rec_items)
    rec_freqs = np.array([item_counter.get(i, 0) for i in range(n_items)])

    indices = np.arange(1, n_items + 1)
    rec_freqs = np.sort(rec_freqs, kind="merge")

    return np.sum((2 * indices - n_items - 1) * rec_freqs) / (
        n_items * np.sum(rec_freqs)
    )


def calc_catalog_coverage_at_k(
    rec_items: np.ndarray,
    n_items: int,
) -> float:
    """calculate catalog coverage at k

    Args:
    - rec_items (np.ndarray): recommended items at k
    - n_items (int): number of items

    Returns:
        float: catalog coverage at k
    """

    return len(set(rec_items)) / n_items


metric_candidates: Dict[str, Callable] = {
    # quantitive metrics
    "Recall": calc_recall_at_k,
    "MAP": calc_average_precision_at_k,
    "DCG": calc_dcg_at_k,
    # qualitative metrics
    "ME": return_exposure_at_k,
    "CatalogCoverage": calc_catalog_coverage_at_k,
    "Gini": calc_gini_at_k,
}
