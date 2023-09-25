from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BaseLoader(ABC):
    @abstractmethod
    def load(self, *args, **kwargs):
        pass
