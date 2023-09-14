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
        trains: Tuple[np.ndarray, np.ndarray],
        vals: Tuple[np.ndarray, np.ndarray],
        test: np.ndarray,
    ) -> list:
        train, train_pscores = trains
        val, val_pscores = vals

        test_pscores = np.ones(len(test))

        self.b = np.mean(train[:, 2])

        train_loss, val_loss, test_loss = [], [], []
        for _ in tqdm(range(self.n_epochs)):
            batch_train, batch_pscores = resample(
                train,
                train_pscores,
                replace=True,
                n_samples=self.batch_size,
                random_state=self.seed,
            )
            for rows, pscore in zip(batch_train, batch_pscores):
                user_id, item_id, click = rows
                err = (click / pscore) - self._predict_pair(user_id, item_id)

                grad_P = -err * self.Q(item_id) + self.reg * self.P(user_id)
                self.P.update(grad=grad_P, index=user_id)

                grad_Q = -err * self.P(user_id) + self.reg * self.Q(item_id)
                self.Q.update(grad=grad_Q, index=item_id)

                grad_b_u = -err + self.reg * self.b_u(user_id)
                self.b_u.update(grad=grad_b_u, index=user_id)

                grad_b_i = -err + self.reg * self.b_i(item_id)
                self.b_i.update(grad=grad_b_i, index=item_id)

            trainloss = self._cross_entropy_loss(
                user_ids=batch_train[:, 0],
                item_ids=batch_train[:, 1],
                clicks=batch_train[:, 2],
                pscores=batch_pscores,
            )
            train_loss.append(trainloss)

            valloss = self._cross_entropy_loss(
                user_ids=val[:, 0],
                item_ids=val[:, 1],
                clicks=val[:, 2],
                pscores=val_pscores,
            )
            val_loss.append(valloss)

            testloss = self._cross_entropy_loss(
                user_ids=test[:, 0],
                item_ids=test[:, 1],
                clicks=test[:, 2],
                pscores=test_pscores,
            )
            test_loss.append(testloss)

        return train_loss, val_loss, test_loss

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
            (clicks / pscores) * np.log(pred_scores)
            + (1 - clicks / pscores) * np.log(1 - pred_scores)
        ) / len(clicks)
        return loss
