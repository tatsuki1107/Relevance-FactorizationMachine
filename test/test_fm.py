# Standard library imports
from abc import ABC


class ModelTestBase(ABC):
    def setup_method(self):
        pass


class TestFM(ModelTestBase):
    def setup_method(self):
        super().setup_method()
        pass

    def test_fit(self):
        pass
