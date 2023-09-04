import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass
from src.mf import ProbabilisticMatrixFactorization as PMF
from src.fm import FactorizationMachine as FM
from utils.metrics import (
    calc_precision_at_k,
    calc_recall_at_k,
    calc_dcg_at_k,
    calc_roc_auc,
)


@dataclass
class Evaluator:
    X: np.ndarray
    y_true: np.ndarray
    indices_per_user: np.ndarray
    K: Tuple[int] = (1, 3, 5)

    def evaluate(self, model: Union[PMF, FM]):
        recall_at_k, precision_at_k, dcg_at_k = [], [], []
        for k in self.K:
            recall_per_user, precision_per_user, dcg_per_user = [], [], []
            for indices in self.indices_per_user:
                y_scores = self._predict(model, indices)
                y_true = self.y_true[indices]
                recall_per_user.append(calc_recall_at_k(y_true, y_scores, k))
                precision_per_user.append(
                    calc_precision_at_k(y_true, y_scores, k)
                )
                dcg_per_user.append(calc_dcg_at_k(y_true, y_scores, k))

            recall_at_k.append(np.mean(recall_per_user))
            precision_at_k.append(np.mean(precision_per_user))
            dcg_at_k.append(np.mean(dcg_per_user))

        thetahold = np.arange(0, 1.0001, 0.0001)[::-1]
        y_scores = self._predict(model)

        tpr, fpr, roc = calc_roc_auc(self.y_true, y_scores, thetahold)
        auc = (fpr, tpr, roc)

        return recall_at_k, precision_at_k, dcg_at_k, auc

    def _predict(
        self, model: Union[PMF, FM], indices: list = None
    ) -> np.ndarray:
        if indices is None:
            indices = np.arange(len(self.X))

        if model.__class__ == FM:
            y_scores = model.predict(self.X[indices])
        elif model.__class__ == PMF:
            user_ids, item_ids = self.X[indices][:, 0], self.X[indices][:, 1]
            y_scores = model.predict(user_ids, item_ids)

        return y_scores
