# Standard library imports
from dataclasses import dataclass
from typing import Dict, Union, Optional
from tqdm import tqdm

# Third-party library imports
import numpy as np
from sklearn.utils import resample
from scipy.sparse import csr_matrix, diags

# Internal modules imports
from src.base import PointwiseBaseRecommender
from utils.optimizer import SGD
from utils.evaluate import ValEvaluator


@dataclass
class FactorizationMachines(PointwiseBaseRecommender):
    """Factorization Machines (FM) model

    Args:
    - n_features (int): The number of features.
    - alpha (float): The scaling factor for the random initialization of the weights.
    - evaluator (Optional[ValEvaluator]): evaluator for the model
    """

    n_features: int
    alpha: float = 2.0
    evaluator: Optional[ValEvaluator] = None

    def __post_init__(self) -> None:
        """initialize the model's parameters"""

        np.random.seed(self.seed)

        w0 = np.array([0.0])
        self.w0 = SGD(params=w0, lr=self.lr)

        limit = self.alpha * np.sqrt(6 / self.n_features)
        # w ~ U(-limit, limit)
        w = np.random.uniform(low=-limit, high=limit, size=self.n_features)
        self.w = SGD(params=w, lr=self.lr)

        # V ~ U(-limit, limit)
        limit = self.alpha * np.sqrt(6 / self.n_factors)
        V = np.random.uniform(
            low=-limit, high=limit, size=(self.n_features, self.n_factors)
        )
        self.V = SGD(params=V, lr=self.lr)

        if self.evaluator is not None:
            self.val_metrics = []
            self.model_name = "FM"

    def fit(
        self,
        train: Dict[str, Union[csr_matrix, np.ndarray]],
        val: Dict[str, Union[csr_matrix, np.ndarray]],
    ) -> list:
        """fit the model to the training data

        Args:
        - train (Dict[str, Union[csr_matrix, np.ndarray]]): training data
        - val (Dict[str, Union[csr_matrix, np.ndarray]]): validation data

        Returns:
            list: training and validation loss
        """

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
            error = (batch_y / batch_pscores) - self.predict(batch_X)
            # shape: (batch_size,)

            # update w0
            self._update_w0(error)
            # update wi
            self._update_w(error, batch_X)
            # update V
            self._update_V(error, batch_X)

            pred_scores = self.predict(batch_X)
            batch_logloss = self._cross_entropy_loss(
                y_trues=batch_y,
                y_scores=pred_scores,
                pscores=batch_pscores,
            )
            train_loss.append(batch_logloss)

            pred_scores = self.predict(val["features"])
            val_logloss = self._cross_entropy_loss(
                y_trues=val["labels"], y_scores=pred_scores, pscores=val["pscores"]
            )
            val_loss.append(val_logloss)

            if self.evaluator is not None:
                eval_features = self.evaluator.features[self.model_name]
                y_scores = self.predict(X=eval_features)
                metric_value = self.evaluator.evaluate(
                    y_scores=y_scores, estimator=self.estimator
                )
                self.val_metrics.append(metric_value)

        return train_loss, val_loss

    def predict(self, X: csr_matrix) -> np.ndarray:
        """predict the scores for the given data

        Args:
        - X (csr_matrix): The input data.

        Returns:
            np.ndarray: The predicted scores.
        """

        # item two
        linear_out = X.dot(self.w())

        # item three
        term1 = np.sum(X.dot(self.V()) ** 2, axis=1)
        term2 = np.sum((X.power(2)).dot(self.V() ** 2), axis=1)
        factor_out = 0.5 * (term1 - term2)

        pred_y = self._sigmoid(self.w0(0) + linear_out + factor_out)
        return pred_y

    def _update_w0(self, error: np.ndarray) -> None:
        """update w0 parameter. see README.md for more details.

        Args:
        - error (np.ndarray): The residuals between the predicted and true labels.
        """

        w0_grad = -np.sum(error)
        self.w0.update(grad=w0_grad, index=None)

    def _update_w(self, error: np.ndarray, X: csr_matrix) -> None:
        """update w parameter. see README.md for more details.

        Args:
        - error (np.ndarray): The residuals between the predicted and true labels.
        - X (csr_matrix): The input data.
        """

        w_grad = -(diags(error) @ X).sum(axis=0).A.flatten()
        self.w.update(grad=w_grad, index=None)

    def _update_V(self, error: np.ndarray, X: csr_matrix) -> None:
        """update V parameter. see README.md for more details.

        Args:
        - error (np.ndarray): The residuals between the predicted and true labels.
        - X (csr_matrix): The input data.
        """

        # calculate V@X.T
        V_dot_X_T = self.V().T @ X.T  # shape: (n_factors, batch_size)
        X_square = X.power(2)  # shape: (batch_size, n_features)

        # calculate grad_V for each factor
        for f in range(self.V().shape[1]):
            term1_f = X.multiply(
                V_dot_X_T[f, :][:, np.newaxis]
            )  # shape: (batch_size, n_features)

            term2_f = X_square.multiply(
                self.V()[:, f]
            )  # shape: (batch_size, n_features)

            grad_V_f = -(
                (term1_f - term2_f).multiply(error[:, None]).sum(axis=0)
            )  # shape: (1, n_features)

            # update V for each factor
            non_zero_feature_indices = grad_V_f.nonzero()[1]
            self.V.update(
                grad=grad_V_f[0, non_zero_feature_indices].A1,
                index=(non_zero_feature_indices, f),
            )
