# Standard library imports
from dataclasses import dataclass
from typing import Tuple, Union
from tqdm import tqdm

# Third-party library imports
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix, diags

# Internal modules imports
from src.base import PointwiseBaseRecommender
from utils.optimizer import SGD


@dataclass
class FactorizationMachines(PointwiseBaseRecommender):
    """Factorization Machines for recommendation.

    Args:
    - n_features (int): 特徴量の次元数
    """

    n_features: int
    alpha: float = 2.0

    def __post_init__(self) -> None:
        """モデルのパラメータを初期化
        - w0: バイアス項
        - w: 線形項の重み
        - V: 交互作用項の重み
        """
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

    def fit(
        self,
        train: Tuple[csr_matrix, np.ndarray, np.ndarray],
        val: Tuple[csr_matrix, np.ndarray, np.ndarray],
    ) -> list:
        """モデルの学習を実行するメソッド

        Args:
        - train (Tuple[特徴量(スパース行列), ラベル, 傾向スコア]): 学習データ.
        - val (tuple[特徴量(スパース行列), ラベル, 傾向スコア]): 検証データ.

        Returns:
        - list: 学習データと検証データのエポック毎の損失.
        """

        train_X, train_y, train_pscores = train
        val_X, val_y, val_pscores = val

        batch_data = self._get_batch_data(
            train_X.copy(),
            train_y.copy(),
            train_pscores.copy(),
        )
        train_loss, val_loss = [], []
        for _ in tqdm(range(self.n_epochs)):
            batch_loss = []
            for batch_X, batch_y, batch_pscores in zip(*batch_data):
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
                batch_loss.append(batch_logloss)

            train_loss.append((np.mean(batch_loss), np.std(batch_loss)))

            pred_scores = self.predict(val_X)
            val_logloss = self._cross_entropy_loss(
                y_trues=val_y, y_scores=pred_scores, pscores=val_pscores
            )
            val_loss.append(val_logloss)

        return train_loss, val_loss

    def predict(self, X: csr_matrix) -> np.ndarray:
        """特徴量から予測確率を計算する

        Args:
        - X (csr_matrix): 特徴量(スパース行列)の配列

        Returns:
        - pred_y (np.ndarray): 予測確率の配列
        """

        # 2項目
        linear_out = X.dot(self.w())

        # 3項目
        term1 = np.sum(X.dot(self.V()) ** 2, axis=1)
        term2 = np.sum((X.power(2)).dot(self.V() ** 2), axis=1)
        factor_out = 0.5 * (term1 - term2)

        pred_y = self._sigmoid(self.w0(0) + linear_out + factor_out)
        return pred_y

    def _update_w0(self, error: np.ndarray) -> None:
        """w0 (バイアス項) を更新する. 更新の詳細は、README.mdを参照

        Args:
        - error (np.ndarray): 予測値と正解ラベルの残差の配列
        """

        w0_grad = -np.sum(error)
        self.w0.update(grad=w0_grad, index=None)

    def _update_w(self, error: np.ndarray, X: csr_matrix) -> None:
        """w (線形項の重み) を更新する. 更新の詳細は、README.mdを参照

        Args:
        - error (np.ndarray): 予測値と正解ラベルの残差の配列
        - X (csr_matrix): 特徴量(スパース行列)の配列
        """

        w_grad = -(diags(error) @ X).sum(axis=0).A.flatten()
        self.w.update(grad=w_grad, index=None)

    def _update_V(self, error: np.ndarray, X: csr_matrix) -> None:
        """V (交互作用項の重み) を更新する. 更新の詳細は、README.mdを参照

        Args:
        - error (np.ndarray): 予測値と正解ラベルの残差の配列
        - X (csr_matrix): 特徴量(スパース行列)の配列
        """

        # 事前に計算できる部分を計算
        V_dot_X_T = self.V().T @ X.T  # shape: (n_factors, batch_size)
        X_square = X.power(2)  # shape: (batch_size, n_features)

        # 各factorごとに計算
        for f in range(self.V().shape[1]):
            # V[:,factor]@X.T * X[:,feature] の計算
            term1_f = X.multiply(
                V_dot_X_T[f, :][:, np.newaxis]
            )  # shape: (batch_size, n_features)

            # V[feature, factor]*X[:,feature]**2 の計算
            term2_f = X_square.multiply(
                self.V()[:, f]
            )  # shape: (batch_size, n_features)

            # error[:, np.newaxis] * (term1_f - term2_f) の計算
            grad_V_f = -(
                (term1_f - term2_f).multiply(error[:, None]).sum(axis=0)
            )  # shape: (1, n_features)

            # update V
            # Vの該当する列（factor）を更新
            non_zero_feature_indices = grad_V_f.nonzero()[1]
            self.V.update(
                grad=grad_V_f[0, non_zero_feature_indices].A1,
                index=(non_zero_feature_indices, f),
            )

    def _get_batch_data(
        self,
        train_X: csr_matrix,
        train_y: np.ndarray,
        train_pscores: np.ndarray,
    ) -> Tuple[list, list, list]:
        """学習データをバッチサイズ毎に分割する

        Args:
            train_X (csr_matrix): 特徴量(スパース行列)の配列
            train_y (np.ndarray): 正解データの配列
            train_pscores (np.ndarray): 傾向スコアの配列

        Returns:
            Tuple[list, list, list]: バッチサイズ毎に分割した学習データ
        """
        train_X, train_y, train_pscores = shuffle(
            train_X, train_y, train_pscores, random_state=self.seed
        )
        batch_X = self._split_data(train_X)
        batch_y = self._split_data(train_y)
        batch_pscores = self._split_data(train_pscores)
        return batch_X, batch_y, batch_pscores

    def _split_data(self, data: Union[csr_matrix, np.ndarray]) -> list:
        """データをバッチサイズ毎に分割する

        Args:
            data (Union[csr_matrix, np.ndarray]): 学習データ

        Returns:
            list: バッチサイズ毎に分割した学習データ
        """
        return [
            data[i: i + self.batch_size]
            for i in range(0, data.shape[0], self.batch_size)
        ]
