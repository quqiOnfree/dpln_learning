from scipy.stats import norm
import numpy as np
from dpln_template import *
from dpln_initilizer import HeInitilizer

class MulLayer(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        self.x = args[0]
        self.y = args[1]
        return args[0] * args[1]

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args[0] + args[1]

    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy
    
class Linear(BaseLayer):
    def __init__(self, input_num: int, output_num: int, bias=True,
                 initilizer: BaseInitilizer = HeInitilizer()):
        super().__init__()
        self.initilizer = initilizer
        self.initilizer.set_shape((input_num, output_num))
        self.W = self.initilizer.generate()
        self.bias = bias
        if bias:
            self.B = np.zeros((1, output_num))

    def forward(self, *args):
        x = args[0]
        self.x = x
        res = x@self.W
        if self.bias:
            res += self.B
        return res

    def backward(self, dout):
        dx = dout @ np.transpose(self.W)
        dW = np.transpose(self.x) @ dout
        dB = np.sum(dout, axis=0)
        self.optimizer_w.update(self.W, dW)
        self.optimizer_b.update(self.B, dB)
        return dx

    def set_optimizer(self, optimizer: BaseOptimizer):
        self.optimizer_w = optimizer.copy()
        self.optimizer_w.set_shape(self.W.shape)
        if self.bias:
            self.optimizer_b = optimizer.copy()
            self.optimizer_b.set_shape(self.B.shape)

class Sigmoid(BaseLayer):
    def forward(self, *args):
        x = args[0]
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1 - self.y)

class Relu(BaseLayer):
    def forward(self, *args):
        x = args[0]
        self.x = x
        self.y = np.maximum(0, x)
        return self.y

    def backward(self, dout):
        return dout * (self.x > 0)

class Gelu(BaseLayer):
    def forward(self, *args):
        x = args[0]
        self.x = x
        return x * norm.cdf(x)

    def backward(self, dout):
        return dout * (norm.cdf(self.x)+self.x*norm.cdf(self.x))

class Softmax(BaseLayer):
    def forward(self, *args):
        x = args[0]
        c = np.max(x, axis=1, keepdims=True)
        self.e = np.exp(x - c)
        self.s = np.sum(self.e, axis=1, keepdims=True)
        return self.e / self.s

    def backward(self, dout):
        s = self.e / self.s
        dot = np.sum(dout * s, axis=1, keepdims=True)
        return s * (dout - dot)

class CrossEntropyError(BaseLayer):
    def __init__(self):
        self.softmax = Softmax()
    
    def loss(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1,t.size)
            t = t.reshape(1,x.size)
    
        self.batch_size = x.shape[0]
        tmp = self.softmax.forward(x)
        self.xt = tmp
        self.t = t
        batch_size = self.t.shape[0]
        return -np.sum(t*np.log(tmp+1e-7), axis=1)/batch_size

    def forward(self, *args):
        return self.loss(args[0], args[1])

    def backward(self, dout = np.array([1.0])):
        batch_size = self.t.shape[0]
        dx = (self.xt-self.t)/batch_size
        return dx
