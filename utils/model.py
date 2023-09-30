from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class FMDataset:
    train: csr_matrix
    val: csr_matrix
    test: csr_matrix


@dataclass
class MFDataset:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    n_users: int
    n_items: int
