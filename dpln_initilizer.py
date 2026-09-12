from dpln_template import BaseInitilizer
import numpy as np

class NormInitilizer(BaseInitilizer):
    def set_shape(self, shape: tuple[int, int]):
        self.shape = shape

    def generate(self):
        return np.random.randn(*self.shape)

class XavierInitilizer(BaseInitilizer):
    """
    Xavier is usually used at sigmoid, tanh functions
    """

    def set_shape(self, shape: tuple[int, int]):
            self.shape = shape

    def generate(self):
        return np.random.randn(*self.shape)/(np.sqrt(self.shape[0])+1e-7)

class HeInitilizer(BaseInitilizer):
    """
    He is usually used at relu, gelu functions
    """

    def set_shape(self, shape: tuple[int, int]):
                self.shape = shape
    
    def generate(self):
        return np.random.randn(*self.shape)/(np.sqrt(self.shape[0]/2.0)+1e-7)
