from dpln_template import BaseOptimizer
import numpy as np

class SGD(BaseOptimizer):
    def __init__(self, alpha: float = 0.003) -> None:
        super().__init__()
        self.alpha = alpha

    def set_shape(self, shape):
        pass

    def copy(self):
        return SGD(self.alpha)

    def update(self, var, dout):
        var -= self.alpha*dout

class Momentum(BaseOptimizer):
    def __init__(self, alpha: float = 0.003) -> None:
        super().__init__()
        self.alpha = alpha

    def set_shape(self, shape):
        self.V = np.zeros(shape)
    
    def copy(self):
        return Momentum(self.alpha)

    def update(self, var, dout):
        self.V = self.alpha * self.V - dout
        var += self.alpha * self.V

class AdaGrad(BaseOptimizer):
    def __init__(self, alpha: float = 0.003) -> None:
        super().__init__()
        self.alpha = alpha

    def set_shape(self, shape):
        self.H = np.zeros(shape)
        
    def copy(self):
        return AdaGrad(self.alpha)

    def update(self, var, dout):
        self.H += self.H + dout * dout
        var -= self.alpha * dout / (np.sqrt(self.H) + 1e-7)

class Adam(BaseOptimizer):
    def __init__(self, alpha=0.003, beta1=0.9, beta2=0.999) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0

    def set_shape(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        
    def copy(self):
        return Adam(self.alpha)

    def update(self, var, dout):
        self.iter += 1
        alpha_t  = self.alpha * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        self.m += (1 - self.beta1) * (dout - self.m)
        self.v += (1 - self.beta2) * (dout**2 - self.v)
        var -= alpha_t * self.m / (np.sqrt(self.v) + 1e-7)
