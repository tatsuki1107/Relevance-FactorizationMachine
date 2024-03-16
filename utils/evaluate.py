# Standard library imports
from typing import Tuple, Dict, Union
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Internal modules imports
from utils.metrics import metric_candidates
from utils.metrics import calc_ips_of_dcg_at_k

METRIC_NAME_ERROR_MESSAGE = "metric_name must be in {}. metric_name: '{}'"
MODEL_CLASS_ERROR_MESSAGE = "model must be FM, MF or Random. model: '{}'"


@dataclass
class _BaseEvaluator(ABC):
    """base class for evaluators

    Args:
    - interaction_df (pd.DataFrame): interaction data
    - features (Dict[str, Union[np.ndarray, csr_matrix]]): features for evaluation.
    """

    interaction_df: pd.DataFrame
    features: Dict[str, Union[np.ndarray, csr_matrix]]

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def _group_by_user_data(self, *args, **kwargs):
        pass


@dataclass
class TestEvaluator(_BaseEvaluator):
    """evaluator for test data

    Args:
    - K (Tuple[int]): ranking positions
    - used_metrics (set): metrics to be used
    - n_items (int): number of items

    Raises:
        ValueError: if metric_name is not included in the set of available metrics
    """

    K: Tuple[int]
    used_metrics: set
    n_items: int

    def __post_init__(self) -> None:
        """store metric functions in a dictionary

        Raises:
            ValueError: if metric_name is not included in the set of available metrics
        """

        self.metric_functions = {}

        # default
        metric_name = "ME"
        self.metric_functions[metric_name] = metric_candidates[metric_name]

        for metric_name in self.used_metrics:
            if metric_name not in metric_candidates:
                raise ValueError(
                    METRIC_NAME_ERROR_MESSAGE.format(
                        metric_candidates.keys(), metric_name
                    )
                )
            self.metric_functions[metric_name] = metric_candidates[metric_name]

    def evaluate(self, y_scores: np.ndarray) -> defaultdict:
        """evaluate the model

        Args:
        - y_scores (np.ndarray): predicted scores

        Returns:
            defaultdict: evaluation results
        """

        group_by_user_data = self._group_by_user_data(y_scores)
        metric_per_user = defaultdict(lambda: defaultdict(list))
        for user, data in group_by_user_data.items():
            ranked_indices = data["y_scores"].argsort()[::-1]
            sorted_y_true = data["labels"][ranked_indices]
            sorted_pscores = data["pscores"][ranked_indices]
            sorted_items = data["items"][ranked_indices]

            if np.sum(sorted_y_true) == 0:
                continue

            for k in self.K:
                for metric_name, metric_func in self.metric_functions.items():
                    if metric_name in {"CatalogCoverage", "Gini"}:
                        metric_per_user[metric_name][k].extend(sorted_items[:k])
                    elif metric_name == "ME":
                        metric_value = metric_func(sorted_pscores, k)
                        metric_per_user[metric_name][k].append(metric_value)
                    else:
                        metric_value = metric_func(sorted_y_true, k)
                        metric_per_user[metric_name][k].append(metric_value)

        results = defaultdict(list)
        for k in self.K:
            for metric_name in self.metric_functions.keys():
                if metric_name in {"CatalogCoverage", "Gini"}:
                    results[metric_name].append(
                        self.metric_functions[metric_name](
                            metric_per_user[metric_name][k], self.n_items
                        )
                    )

                else:
                    results[metric_name].append(
                        np.nanmean(metric_per_user[metric_name][k])
                    )

        return results

    def _group_by_user_data(
        self, y_scores: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """group the data by user

        Args:
        - y_scores (np.ndarray): predicted scores

        Returns:
            Dict[str, Dict[str, np.ndarray]]: grouped data
        """

        self.interaction_df["y_score"] = y_scores
        return (
            self.interaction_df.groupby("user")
            .agg(list)
            .map(np.array)
            .apply(
                lambda row: {
                    "items": row["item"],
                    "labels": row["label"],
                    "y_scores": row["y_score"],
                    "pscores": row["pscore"],
                },
                axis=1,
            )
            .to_dict()
        )


@dataclass
class ValEvaluator(_BaseEvaluator):
    """evaluator for validation data

    Args:
    - k (int): ranking position
    - metric_name (str): metric name
    """

    k: int
    metric_name: str

    def __post_init__(self) -> None:
        """store metric function

        Raises:
            ValueError: if metric_name is not DCG
        """

        if self.metric_name == "DCG":
            self.metric_func = calc_ips_of_dcg_at_k
        else:
            raise ValueError("You can use only DCG metric.")

    def evaluate(self, y_scores: np.ndarray, estimator: str) -> float:
        """evaluate the model

        Args:
        - y_scores (np.ndarray): predicted scores
        - estimator (str): IPS or Naive estimator

        Returns:
            float: evaluation result
        """

        group_by_user_data = self._group_by_user_data(y_scores, estimator)
        metric_per_user = []
        for user, data in group_by_user_data.items():
            ranked_indices = data["y_scores"].argsort()[::-1]
            sorted_y_true = data["labels"][ranked_indices]
            sorted_pscores = data["pscores"][ranked_indices]

            if np.sum(sorted_y_true) == 0:
                continue

            metric_value = self.metric_func(sorted_y_true, self.k, sorted_pscores)
            metric_per_user.append(metric_value)

        return np.mean(metric_per_user)

    def _group_by_user_data(
        self, y_scores: np.ndarray, estimator: str
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """group the data by user

        Args:
        - y_scores (np.ndarray): predicted scores
        - estimator (str): IPS or Naive estimator

        Returns:
            Dict[str, Dict[str, np.ndarray]]: grouped data
        """

        self.interaction_df["y_score"] = y_scores
        pscore_name = "pscore" if estimator == "IPS" else "ones_pscore"

        return (
            self.interaction_df.groupby("user")
            .agg(list)
            .map(np.array)
            .apply(
                lambda row: {
                    "items": row["item"],
                    "labels": row["label"],
                    "y_scores": row["y_score"],
                    "pscores": row[pscore_name],
                },
                axis=1,
            )
            .to_dict()
        )
