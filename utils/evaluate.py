import numpy as np
from typing import Tuple, Union, Optional
from collections import defaultdict
from dataclasses import dataclass
from src.mf import ProbabilisticMatrixFactorization as PMF
from src.fm import FactorizationMachine as FM
from utils.metrics import metrics


@dataclass
class Evaluator:
    X: np.ndarray
    y_true: np.ndarray
    indices_per_user: np.ndarray
    used_metrics: set
    K: Tuple[int] = (1, 3, 5)

    def __post_init__(self) -> None:
        for metric_name in set(metrics.keys()):
            if metric_name not in self.used_metrics:
                metrics.pop(metric_name)

    def evaluate(
        self, model: Union[PMF, FM], pscores: Optional[np.ndarray] = None
    ) -> defaultdict:
        if pscores is None:
            pscores = np.ones_like(self.y_true)

        results = defaultdict(list)
        metric_per_user = defaultdict(lambda: defaultdict(list))
        for indices in self.indices_per_user:
            y_scores = self._predict(model, indices)
            y_true = self.y_true[indices]
            user_pscores = pscores[indices]

            for k in self.K:
                for metric_name, metric_func in metrics.items():
                    if metric_name == "ROC_AUC":
                        continue
                    metric_per_user[metric_name][k].append(
                        metric_func(y_true, y_scores, k, user_pscores)
                    )

        for k in self.K:
            for metric_name in metrics.keys():
                results[metric_name].append(
                    np.nanmean(metric_per_user[metric_name][k])
                )

        if "ROC_AUC" in metrics:
            metric_func = metrics["ROC_AUC"]
            thetahold = np.arange(0, 1.0001, 0.0001)[::-1]
            y_scores = self._predict(model)

            tpr, fpr, roc = metric_func(self.y_true, y_scores, thetahold)
            auc = (fpr, tpr, roc)
            results["ROC_AUC"].append(auc)

        return results

    def _predict(
        self,
        model: Union[PMF, FM],
        indices: list = None,
    ) -> np.ndarray:
        if indices is None:
            indices = np.arange(len(self.y_true))

        if model.__class__ == FM:
            y_scores = model.predict(self.X[indices])
        elif model.__class__ == PMF:
            user_ids, item_ids = self.X[indices][:, 0], self.X[indices][:, 1]
            y_scores = model.predict(user_ids, item_ids)
        else:
            y_scores = np.random.randint(0, 2, len(indices))

        return y_scores
