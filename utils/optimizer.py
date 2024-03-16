# Standard library imports
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Optional

# Third-party library imports
import numpy as np


@dataclass
class BaseOptimizer(ABC):
    """base class for optimizers

    Args:
    - params (np.ndarray): parameters to be optimized
    - lr (float): learning rate
    """

    params: np.ndarray
    lr: float

    @abstractmethod
    def update(
        self,
        grad: Union[float, np.ndarray],
        index: Optional[Union[int, tuple]],
    ) -> None:
        """Update the parameters using the given gradient.

        Args:
        - grad (Union[float, np.ndarray]): Gradient of the parameters.
        - index (Optional[Union[int, tuple]]): Index of the parameters to be updated.
        """
        pass

    def __call__(self, index=None) -> Union[np.ndarray, float]:
        """Get the current parameters or a specific parameter value.

        Args:
        - index (Optional[int, tuple]): Index of the parameters to be returned.

        Returns:
        - (Union[np.ndarray, float]): Parameters or a specific parameter value.
        """

        if index is None:
            return self.params

        return self.params[index]


@dataclass
class SGD(BaseOptimizer):
    """Stochastic Gradient Descent optimizer."""

    def update(
        self,
        grad: Union[float, np.ndarray],
        index: Optional[Union[int, tuple]],
    ) -> None:
        if index is None:
            self.params -= self.lr * grad
        else:
            self.params[index] -= self.lr * grad
