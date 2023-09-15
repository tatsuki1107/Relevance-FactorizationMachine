import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from tqdm import tqdm
from sklearn.utils import resample
from scipy.sparse import csr_matrix, diags
from src.base import PointwiseBaseRecommender


@dataclass
class FactorizationMachine(PointwiseBaseRecommender):
    n_features: int

    def __post_init__(self) -> None:
        np.random.seed(self.seed)

        self.w0 = 0.0
        self.w = np.zeros(self.n_features)

        self.V = np.random.normal(
            scale=self.scale,
            size=(self.n_features, self.n_factors),
        )

    def fit(
        self,
        train: Tuple[csr_matrix, np.ndarray, np.ndarray],
        val: Tuple[csr_matrix, np.ndarray, np.ndarray],
        test: Tuple[csr_matrix, np.ndarray],
    ) -> list:
        train_X, train_y, train_pscores = train
        val_X, val_y, val_pscores = val
        test_X, test_y = test

        train_loss, val_loss, test_loss = [], [], []
        for _ in tqdm(range(self.n_epochs)):
            batch_X, batch_y, batch_pscores = resample(
                train_X,
                train_y,
                train_pscores,
                replace=True,
                n_samples=self.batch_size,
                random_state=self.seed,
            )
            error = (batch_y / batch_pscores) - self.predict(batch_X)
            # shape: (batch_size,)

            # update w0
            self._update_w0(error)
            # update wi
            self._update_w(error, batch_X)
            # update V
            self._update_V(error, batch_X)

            trainloss = self._cross_entropy_loss(
                X=batch_X, y=batch_y, pscores=batch_pscores
            )
            train_loss.append(trainloss)

            valloss = self._cross_entropy_loss(
                X=val_X, y=val_y, pscores=val_pscores
            )
            val_loss.append(valloss)

            testloss = self._cross_entropy_loss(
                X=test_X, y=test_y, pscores=None
            )
            test_loss.append(testloss)

        return train_loss, val_loss, test_loss

    def predict(self, X: csr_matrix) -> np.ndarray:
        # 2項目
        linear_out = X.dot(self.w)

        # 3項目
        term1 = np.sum(X.dot(self.V) ** 2, axis=1)
        term2 = np.sum((X.power(2)).dot(self.V**2), axis=1)
        factor_out = 0.5 * (term1 - term2)

        return self._sigmoid(self.w0 + linear_out + factor_out)

    def _cross_entropy_loss(
        self,
        X: csr_matrix,
        y: np.ndarray,
        pscores: Optional[np.ndarray] = None,
    ) -> float:
        if pscores is None:
            pscores = np.ones_like(y)

        y_hat = self.predict(X)
        loss = -np.sum(
            (y / pscores) * np.log(y_hat)
            + (1 - (y / pscores)) * np.log(1 - y_hat)
        ) / len(y)
        return loss

    def _update_w0(self, error: np.ndarray) -> None:
        """update w0"""
        w0_grad = -np.sum(error)
        self.w0 -= self.lr * w0_grad

    def _update_w(self, error: np.ndarray, X: csr_matrix) -> None:
        """update wi"""
        w_grad = -(diags(error) @ X).sum(axis=0).A.flatten()
        self.w -= self.lr * w_grad

    def _update_V(self, error: np.ndarray, X: csr_matrix) -> None:
        """update V"""
        # 事前に計算できる部分を計算
        V_dot_X_T = self.V.T @ X.T  # shape: (n_factors, batch_size)
        X_square = X.power(2)  # shape: (batch_size, n_features)

        # 各factorごとに計算
        for f in range(self.V.shape[1]):
            # V[:,factor]@X.T * X[:,feature] の計算
            term1_f = X.multiply(
                V_dot_X_T[f, :][:, np.newaxis]
            )  # shape: (batch_size, n_features)

            # V[feature, factor]*X[:,feature]**2 の計算
            term2_f = X_square.multiply(
                self.V[:, f]
            )  # shape: (batch_size, n_features)

            # error[:, np.newaxis] * (term1_f - term2_f) の計算
            grad_V_f = -(
                (term1_f - term2_f).multiply(error[:, None]).sum(axis=0)
            )  # shape: (1, n_features)
            # Vの該当する列（factor）を更新
            non_zero_feature_indices = grad_V_f.nonzero()[1]

            # update V
            self.V[non_zero_feature_indices, f] -= (
                self.lr * grad_V_f[0, non_zero_feature_indices].A1
            )
