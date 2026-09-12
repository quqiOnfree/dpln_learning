from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Self

class BaseOptimizer(ABC):
    @abstractmethod
    def set_shape(self, shape: tuple[int, int]):
        pass

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def update(self, var: np.ndarray, dout: np.ndarray):
        pass

class BaseLayer(ABC):
    @abstractmethod
    def forward(self, *args: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> Any:
        pass

    def set_optimizer(self, optimizer: BaseOptimizer):
        pass

class BaseInitilizer:
    @abstractmethod
    def set_shape(self, shape: tuple[int, int]):
        pass

    @abstractmethod
    def generate(self) -> np.ndarray:
        pass

    def __call__(self) -> np.ndarray:
        return self.generate()
