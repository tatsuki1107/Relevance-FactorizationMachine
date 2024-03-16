# Standard library imports
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BaseLoader(ABC):
    """base class for data loaders"""

    @abstractmethod
    def load(self, *args, **kwargs):
        pass
