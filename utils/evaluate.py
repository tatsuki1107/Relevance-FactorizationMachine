import numpy as np
from typing import Tuple, Union, Optional
from collections import defaultdict
from dataclasses import dataclass
from src.mf import LogisticMatrixFactorization as MF
from src.fm import FactorizationMachine as FM
from utils.metrics import metrics


@dataclass
class Evaluator:
    X: Optional[np.ndarray]
    y_true: np.ndarray
    indices_per_user: np.ndarray
    used_metrics: set
    K: Tuple[int] = (1, 3, 5)
    thetahold: Optional[float] = None

    def __post_init__(self) -> None:
        self.metrics = {}
        for metric_name in self.used_metrics:
            if metric_name not in metrics:
                raise ValueError(
                    f"metric_name must be in {metrics.keys()}. "
                    + f"metric_name: {metric_name}"
                )
            self.metrics[metric_name] = metrics[metric_name]

    def evaluate(
        self, model: Union[MF, FM, str], pscores: Optional[np.ndarray] = None
    ) -> defaultdict:
        if pscores is None:
            pscores = np.ones_like(self.y_true)

        metric_per_user = defaultdict(lambda: defaultdict(list))
        for indices in self.indices_per_user:
            y_scores = self._predict(model, indices)
            y_true = self.y_true[indices]
            user_pscores = pscores[indices]

            for k in self.K:
                for metric_name, metric_func in self.metrics.items():
                    if metric_name == "ROC_AUC":
                        continue
                    metric_per_user[metric_name][k].append(
                        metric_func(y_true, y_scores, k, user_pscores)
                    )

        results = defaultdict(list)
        for k in self.K:
            for metric_name in self.metrics.keys():
                results[metric_name].append(
                    np.nanmean(metric_per_user[metric_name][k])
                )

        return results

    def _predict(
        self,
        model: Union[MF, FM, str],
        indices: list = None,
    ) -> np.ndarray:
        if indices is None:
            indices = np.arange(len(self.y_true))

        if model.__class__ == FM:
            y_scores = model.predict(self.X[indices])
        elif model.__class__ == MF:
            user_ids, item_ids = self.X[indices][:, 0], self.X[indices][:, 1]
            y_scores = model.predict(user_ids, item_ids)
        elif model == "Random":
            y_scores = np.random.randint(0, 2, len(indices))
        else:
            raise ValueError(
                "model must be FM, PMF or Random. " + f"model: {model}"
            )

        if self.thetahold is None:
            return y_scores

        return (y_scores >= self.thetahold).astype(int)
