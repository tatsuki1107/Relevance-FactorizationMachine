# Standard library imports
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BaseLoader(ABC):
    """データのロードに使う基底クラス"""

    @abstractmethod
    def load(self, *args, **kwargs):
        pass
