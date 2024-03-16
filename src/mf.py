# Standard library imports
from typing import Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Third-party library imports
import numpy as np
from sklearn.utils import resample

# Internal modules imports
from utils.optimizer import SGD
from src.base import PointwiseBaseRecommender
from utils.evaluate import ValEvaluator


@dataclass
class LogisticMatrixFactorization(PointwiseBaseRecommender):
    """Logistic Matrix Factorization (MF) model

    Args:
    - n_users (int): The number of users.
    - n_items (int): The number of items.
    - reg (float): The regularization parameter.
    - alpha (float): The scaling factor for the random initialization of the weights.
    - evaluator (Optional[ValEvaluator]): evaluator for the model
    """

    n_users: int
    n_items: int
    reg: float
    alpha: float = 4.0
    evaluator: Optional[ValEvaluator] = None

    def __post_init__(self) -> None:
        """initialize the model's parameters"""

        np.random.seed(self.seed)

        # init user embeddings
        limit = self.alpha * np.sqrt(6 / self.n_factors)
        # P ~ U(-limit, limit)
        P = np.random.uniform(
            low=-limit, high=limit, size=(self.n_users, self.n_factors)
        )
        self.P = SGD(params=P, lr=self.lr)

        # init item embeddings
        # Q ~ U(-limit, limit)
        Q = np.random.uniform(
            low=-limit, high=limit, size=(self.n_items, self.n_factors)
        )
        self.Q = SGD(params=Q, lr=self.lr)

        # init user bias
        # b_u ~ N(0, 0.001^2)
        b_u = np.random.normal(scale=0.001, size=self.n_users)
        self.b_u = SGD(params=b_u, lr=self.lr)

        # init item bias
        # b_i ~ N(0, 0.001^2)
        b_i = np.random.normal(scale=0.001, size=self.n_items)
        self.b_i = SGD(params=b_i, lr=self.lr)

        if self.evaluator is not None:
            self.val_metrics = []
            self.model_name = "MF"

    def fit(
        self,
        train: Dict[str, np.ndarray],
        val: Dict[str, np.ndarray],
    ) -> list:
        """fit the model to the training data

        Args:
        - train (Dict[str, np.ndarray]): training data
        - val (Dict[str, np.ndarray]): validation data

        Returns:
            list: training loss and validation loss
        """

        # init global bias
        self.b = np.mean(train["labels"])

        train_loss, val_loss = [], []
        for epoch in tqdm(range(self.n_epochs)):
            batch_X, batch_y, batch_pscores = resample(
                train["features"],
                train["labels"],
                train["pscores"],
                replace=False,
                n_samples=self.batch_size,
                random_state=epoch,
            )

            for feature, click, pscore in zip(batch_X, batch_y, batch_pscores):
                user_id, item_id = feature[0], feature[1]
                err = (click / pscore) - self._predict_pair(user_id, item_id)

                # update user embeddings
                self._update_P(user_id=user_id, item_id=item_id, err=err)
                # update item embeddings
                self._update_Q(user_id=user_id, item_id=item_id, err=err)
                # update user bias
                self._update_b_u(user_id=user_id, err=err)
                # update item bias
                self._update_b_i(item_id=item_id, err=err)

            pred_scores = self.predict(batch_X)
            batch_logloss = self._cross_entropy_loss(
                y_trues=batch_y,
                y_scores=pred_scores,
                pscores=batch_pscores,
            )
            train_loss.append(batch_logloss)

            pred_scores = self.predict(val["features"])
            val_logloss = self._cross_entropy_loss(
                y_trues=val["labels"],
                y_scores=pred_scores,
                pscores=val["pscores"],
            )
            val_loss.append(val_logloss)

            if self.evaluator is not None:
                eval_features = self.evaluator.features[self.model_name]
                y_scores = self.predict(eval_features)
                metric_value = self.evaluator.evaluate(
                    y_scores=y_scores, estimator=self.estimator
                )
                self.val_metrics.append(metric_value)

        return train_loss, val_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict the probability of the given samples

        Args:
        - X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted probabilities.
        """

        user_ids, item_ids = X[:, 0], X[:, 1]
        return np.array(
            [
                self._predict_pair(user_id, item_id)
                for user_id, item_id in zip(user_ids, item_ids)
            ]
        )

    def _predict_pair(self, user_id: int, item_id: int) -> float:
        """predict the probability of the given pair of user and item

        Args:
            user_id (int): user ID
            item_id (int): item ID

        Returns:
            float: The predicted probability.
        """

        return self._sigmoid(
            np.dot(self.P(user_id), self.Q(item_id))
            + self.b_u(user_id)
            + self.b_i(item_id)
            + self.b
        )

    def _update_P(self, user_id: int, item_id: int, err: float) -> None:
        """update user embeddings. see README.md for more details.

        Args:
        - user_id (int): user ID
        - item_id (int): item ID
        - err (float): The residual between the predicted value and the true label.
        """

        grad_P = -err * self.Q(item_id) + self.reg * self.P(user_id)
        self.P.update(grad=grad_P, index=user_id)

    def _update_Q(self, user_id: int, item_id: int, err: float) -> None:
        """update item embeddings. see README.md for more details.

        Args:
        - user_id (int): user ID
        - item_id (int): item ID
        - err (float): The residual between the predicted value and the true label.
        """

        grad_Q = -err * self.P(user_id) + self.reg * self.Q(item_id)
        self.Q.update(grad=grad_Q, index=item_id)

    def _update_b_u(self, user_id: int, err: float) -> None:
        """update user bias. see README.md for more details.

        Args:
        - user_id (int): user ID
        - err (float): The residual between the predicted value and the true label.
        """

        grad_b_u = -err + self.reg * self.b_u(user_id)
        self.b_u.update(grad=grad_b_u, index=user_id)

    def _update_b_i(self, item_id: int, err: float) -> None:
        """update item embeddings. see README.md for more details.

        Args:
        - item_id (int): item ID
        - err (float): The residual between the predicted value and the true label.
        """

        grad_b_i = -err + self.reg * self.b_i(item_id)
        self.b_i.update(grad=grad_b_i, index=item_id)
