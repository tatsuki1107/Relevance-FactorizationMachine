import numpy as np
from typing import Tuple
from dataclasses import dataclass
from utils.optimizer import SGD
from tqdm import tqdm
from sklearn.utils import resample
from src.base import PointwiseBaseRecommender


@dataclass
class LogisticMatrixFactorization(PointwiseBaseRecommender):
    """
    Logistic Matrix Factorization model for recommendation.

    Args:
    - n_users (int): ユーザー数
    - n_items (int): アイテム数
    - reg (float): L2正則化項の係数
    - alpha (float, optional): 因子行列のスケール調整パラメータ .Defaults to 2.
    """

    n_users: int
    n_items: int
    reg: float
    alpha: float = 2

    def __post_init__(self) -> None:
        """モデルのパラメータを初期化
        - P: ユーザーの因子行列
        - Q: アイテムの因子行列
        - b_u: ユーザーのバイアス項
        - b_i: アイテムのバイアス項
        """
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

    def fit(
        self,
        train: Tuple[np.ndarray, np.ndarray, np.ndarray],
        val: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> list:
        """
        モデルの学習を実行するメソッド

        Args:
        - train (Tuple[特徴量, ラベル, 傾向スコア]): 学習データ.
        - val (tuple[特徴量, ラベル, 傾向スコア]): 検証データ.

        Returns:
        - list: 学習データと検証データのエポック毎の損失.
        """

        train_X, train_y, train_pscores = train
        val_X, val_y, val_pscores = val

        # init global bias
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

            pred_scores = self.predict(batch_X)
            train_logloss = self._cross_entropy_loss(
                y_trues=batch_y,
                y_scores=pred_scores,
                pscores=batch_pscores,
            )
            train_loss.append(train_logloss)

            pred_scores = self.predict(val_X)
            val_logloss = self._cross_entropy_loss(
                y_trues=val_y,
                y_scores=pred_scores,
                pscores=val_pscores,
            )
            val_loss.append(val_logloss)

        return train_loss, val_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """特徴量(ユーザー、アイテムのID)から予測確率を計算する

        Args:
        - X (np.ndarray): 特徴量の配列

        Returns:
        - (np.ndarray): 予測確率の配列
        """
        user_ids, item_ids = X[:, 0], X[:, 1]
        return np.array(
            [
                self._predict_pair(user_id, item_id)
                for user_id, item_id in zip(user_ids, item_ids)
            ]
        )

    def _predict_pair(self, user_id: int, item_id: int) -> float:
        """ユーザーとアイテムのペアから予測確率を計算する

        Args:
        - user_id (int): 単一のユーザーID
        - item_id (int): 単一のアイテムID

        Returns:
        - (float): 単一の予測確率
        """
        return self._sigmoid(
            np.dot(self.P(user_id), self.Q(item_id))
            + self.b_u(user_id)
            + self.b_i(item_id)
            + self.b
        )

    def _update_P(self, user_id: int, item_id: int, err: float) -> None:
        """ユーザーの因子行列を更新する。更新の詳細は、README.mdを参照

        Args:
            user_id (int): 単一のユーザーID
            item_id (int): 単一のアイテムID
            err (float): 予測値と正解ラベルの残差
        """
        grad_P = -err * self.Q(item_id) + self.reg * self.P(user_id)
        self.P.update(grad=grad_P, index=user_id)

    def _update_Q(self, user_id: int, item_id: int, err: float) -> None:
        """アイテムの因子行列を更新する。更新の詳細は、README.mdを参照

        Args:
            user_id (int): 単一のユーザーID
            item_id (int): 単一のアイテムID
            err (float): 予測値と正解ラベルの残差
        """
        grad_Q = -err * self.P(user_id) + self.reg * self.Q(item_id)
        self.Q.update(grad=grad_Q, index=item_id)

    def _update_b_u(self, user_id: int, err: float) -> None:
        """ユーザーのバイアス項を更新する。更新の詳細は、README.mdを参照

        Args:
            user_id (int): 単一のユーザーID
            err (float): 予測値と正解ラベルの残差
        """
        grad_b_u = -err + self.reg * self.b_u(user_id)
        self.b_u.update(grad=grad_b_u, index=user_id)

    def _update_b_i(self, item_id: int, err: float) -> None:
        """アイテムのバイアス項を更新する。更新の詳細は、README.mdを参照

        Args:
            item_id (int): 単一のアイテムID
            err (float): 予測値と正解ラベルの残差
        """

        grad_b_i = -err + self.reg * self.b_i(item_id)
        self.b_i.update(grad=grad_b_i, index=item_id)
