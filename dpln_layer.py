from scipy.stats import norm
import numpy as np
from dpln_template import *
from dpln_initilizer import HeInitilizer
from dpln_template import BaseOptimizer

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

class BatchNorm(BaseLayer):
    def __init__(self, input_output_num: int) -> None:
        super().__init__()
        self.input_output_num = input_output_num
        self.gamma = np.ones((1, input_output_num))
        self.beta = np.zeros((1, input_output_num))
        self.epsilon = 1e-7

    def forward(self, *args):
        x = args[0]
        self.x = x
        self.mu = 1/self.input_output_num*np.sum(x,axis=1)
        self.centered_x=x-self.mu
        self.var = 1/self.input_output_num*np.sum(self.centered_x**2,axis=1)
        self.std=np.sqrt(self.var)
        self.nonzero_std=self.std+self.epsilon
        self.o_x = self.centered_x/self.nonzero_std
        return self.gamma*self.o_x+self.beta

    def backward(self, dout):
        dbeta=dout.sum(axis=0)
        dgamma=np.sum(dout*self.o_x,axis=0)
        dmu=1/self.input_output_num
        dvar=(2*self.input_output_num-2)/self.input_output_num**2*self.centered_x
        dx=dout*self.gamma*((1-dmu)*self.nonzero_std-dvar/2/self.std*self.centered_x)/self.nonzero_std**2
        self.beta_optimizer.update(self.beta, dbeta)
        self.gamma_optimizer.update(self.gamma,dgamma)
        return dx

    def set_optimizer(self, optimizer: BaseOptimizer):
        self.beta_optimizer=optimizer.copy()
        self.beta_optimizer.set_shape(self.beta.shape)
        self.gamma_optimizer=optimizer.copy()
        self.gamma_optimizer.set_shape(self.gamma.shape)

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
