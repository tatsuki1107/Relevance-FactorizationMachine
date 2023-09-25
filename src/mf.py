import numpy as np
from typing import Tuple
from dataclasses import dataclass
from utils.optimizer import Adam
from tqdm import tqdm
from sklearn.utils import resample
from src.base import PointwiseBaseRecommender


@dataclass
class ProbabilisticMatrixFactorization(PointwiseBaseRecommender):
    n_users: int
    n_items: int
    reg: float
    eps: float = 1e-8

    def __post_init__(self) -> None:
        np.random.seed(self.seed)

        # init user embeddings
        P = np.random.normal(
            scale=self.scale, size=(self.n_users, self.n_factors)
        )
        self.P = Adam(params=P, lr=self.lr)

        # init item embeddings
        Q = np.random.normal(
            scale=self.scale, size=(self.n_items, self.n_factors)
        )
        self.Q = Adam(params=Q, lr=self.lr)

        # init user bias
        b_u = np.zeros(self.n_users)
        self.b_u = Adam(params=b_u, lr=self.lr)

        # init item bias
        b_i = np.zeros(self.n_items)
        self.b_i = Adam(params=b_i, lr=self.lr)

    def fit(
        self,
        train: Tuple[np.ndarray, np.ndarray, np.ndarray],
        val: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> list:
        train
        train_X, train_y, train_pscores = train
        val_X, val_y, val_pscores = val

        self.b = np.mean(train_y)

        train_loss, val_loss = [], []
        for _ in tqdm(range(self.n_epochs)):
            batch_X, batch_y, batch_pscores = resample(
                train_X,
                train_y,
                train_pscores,
                replace=True,
                n_samples=self.batch_size,
                random_state=self.seed,
            )
            for features, click, pscore in zip(
                batch_X, batch_y, batch_pscores
            ):
                user_id, item_id = features[0], features[1]
                err = (click / pscore) - self._predict_pair(user_id, item_id)

                # update user embeddings
                self._update_P(user_id=user_id, item_id=item_id, err=err)
                # update item embeddings
                self._update_Q(user_id=user_id, item_id=item_id, err=err)
                # update user bias
                self._update_b_u(user_id=user_id, err=err)
                # update item bias
                self._update_b_i(item_id=item_id, err=err)

            trainloss = self._cross_entropy_loss(
                user_ids=batch_X[:, 0],
                item_ids=batch_X[:, 1],
                clicks=batch_y,
                pscores=batch_pscores,
            )
            train_loss.append(trainloss)

            valloss = self._cross_entropy_loss(
                user_ids=val_X[:, 0],
                item_ids=val_X[:, 1],
                clicks=val_y,
                pscores=val_pscores,
            )
            val_loss.append(valloss)

        return train_loss, val_loss

    def predict(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [
                self._predict_pair(user_id, item_id)
                for user_id, item_id in zip(user_ids, item_ids)
            ]
        )

    def _predict_pair(self, user_id: int, item_id: int) -> float:
        return self._sigmoid(
            np.dot(self.P(user_id), self.Q(item_id))
            + self.b_u(user_id)
            + self.b_i(item_id)
            + self.b
        )

    def _cross_entropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        clicks: np.ndarray,
        pscores: np.ndarray,
    ) -> float:
        pred_scores = self.predict(user_ids, item_ids)
        loss = -np.sum(
            (clicks / pscores) * np.log(pred_scores + self.eps)
            + (1 - (clicks / pscores)) * np.log(1 - pred_scores + self.eps)
        ) / len(clicks)
        return loss

    def _update_P(self, user_id: int, item_id: int, err: float) -> None:
        grad_P = -err * self.Q(item_id) + self.reg * self.P(user_id)
        self.P.update(grad=grad_P, index=user_id)

    def _update_Q(self, user_id: int, item_id: int, err: float) -> None:
        grad_Q = -err * self.P(user_id) + self.reg * self.Q(item_id)
        self.Q.update(grad=grad_Q, index=item_id)

    def _update_b_u(self, user_id: int, err: float) -> None:
        grad_b_u = -err + self.reg * self.b_u(user_id)
        self.b_u.update(grad=grad_b_u, index=user_id)

    def _update_b_i(self, item_id: int, err: float) -> None:
        grad_b_i = -err + self.reg * self.b_i(item_id)
        self.b_i.update(grad=grad_b_i, index=item_id)
