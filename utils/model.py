from dataclasses import dataclass
import numpy as np
from typing import Union, List, TypeVar
from scipy.sparse import csr_matrix

MatrixType = TypeVar("MatrixType", csr_matrix, np.ndarray)


@dataclass
class Dataset:
    train: Union[List[MatrixType], np.ndarray]
    val: Union[List[MatrixType], np.ndarray]
    test: Union[List[MatrixType], np.ndarray]


@dataclass
class Pscores:
    train: np.ndarray
    val: np.ndarray
