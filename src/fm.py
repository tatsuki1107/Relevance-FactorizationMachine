import numpy as np
from dataclasses import dataclass
from typing import Tuple
from tqdm import tqdm
from sklearn.utils import resample
from scipy.sparse import csr_matrix


@dataclass
class FactorizationMachine:
    n_epochs: int
    n_factors: int
    n_features: int
    scale: float
    lr: float
    seed: int
    batch_size: int

    def __post_init__(self) -> None:
        np.random.seed(self.seed)

        self.w0 = 0.0
        self.w = np.zeros(self.n_features)
        # self.w = Adam(params=w, lr=self.lr)

        self.V = np.random.normal(
            scale=self.scale,
            size=(self.n_features, self.n_factors),
        )
        # self.V = Adam(params=V, lr=self.lr)

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

            # update w0
            w0_grad = -np.sum(error)
            self.w0 -= self.lr * w0_grad

            # update wi
            w_grad = -np.sum(error[:, None] * batch_X, axis=0)
            self.w -= self.lr * w_grad
            # self.w.update(grad=w_grad)

            # 事前に計算できる部分を計算
            V_dot_X_T = self.V.T @ batch_X.T
            X_square = batch_X**2

            # term1: V[:,factor]@X.T * X[:,feature] の計算
            term1 = np.einsum("di,fd->dif", batch_X, V_dot_X_T)

            # term2: V[feature, factor]*X[:,feature]**2 の計算
            term2 = self.V * X_square[:, :, np.newaxis]

            # update V
            grad_V = -np.sum(
                error[:, np.newaxis, np.newaxis] * (term1 - term2), axis=0
            )
            self.V -= self.lr * grad_V
            # self.V.update(grad=grad_V)

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 2項目
        linear_out = np.dot(X, self.w)

        # 3項目
        term1 = np.sum((X @ self.V) ** 2, axis=1)
        term2 = np.sum((X**2) @ (self.V**2), axis=1)
        factor_out = 0.5 * (term1 - term2)

        return self._sigmoid(self.w0 + linear_out + factor_out)

    def _cross_entropy_loss(self, X, y, pscores=None):
        if pscores is None:
            pscores = np.ones_like(y)

        y_hat = self.predict(X)
        loss = -np.sum(
            (y / pscores) * np.log(y_hat)
            + (1 - (y / pscores)) * np.log(1 - y_hat)
        ) / len(y)
        return loss

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
