from dataclasses import dataclass
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix


@dataclass
class Dataset:
    train: Union[csr_matrix, np.ndarray]
    val: Union[csr_matrix, np.ndarray]
    test: Union[csr_matrix, np.ndarray]


@dataclass
class Pscores:
    train: np.ndarray
    val: np.ndarray
