import numpy as np
from dataset.mnist import load_mnist
from dpln_template import BaseLayer
from dpln_layer import *
from dpln_optimizer import *

class MyNet(BaseLayer):
    def __init__(self) -> None:
        self.L1 = Linear(784, 392)
        self.R1 = Relu()
        self.L2 = Linear(392, 196)
        self.R2 = Relu()
        self.L3 = Linear(196, 10)

    def forward(self, *args):
        x = args[0]
        x = self.L1.forward(x)
        x = self.R1.forward(x)
        x = self.L2.forward(x)
        x = self.R2.forward(x)
        x = self.L3.forward(x)
        return x

    def backward(self, dout):
        dout = self.L3.backward(dout)
        dout = self.R2.backward(dout)
        dout = self.L2.backward(dout)
        dout = self.R1.backward(dout)
        dout = self.L1.backward(dout)
        return dout

    def set_optimizer(self, optimizer: BaseOptimizer):
        self.L1.set_optimizer(optimizer)
        self.R1.set_optimizer(optimizer)
        self.L2.set_optimizer(optimizer)
        self.R2.set_optimizer(optimizer)
        self.L3.set_optimizer(optimizer)

if __name__ == "__main__":
    loss = CrossEntropyError()
    mynet = MyNet()
    alpha = 0.001
    mynet.set_optimizer(Adam(alpha))
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

    mloss = 5.0
    stop_time = 0
    for j in range(100):
        for i in range(0, len(t_train), 256):
            x = x_train[i:i+256]
            t = t_train[i:i+256]
            y = mynet.forward(x)
            loss.loss(y, np.eye(10)[t])
            mynet.backward(loss.backward(np.array([1.0])))
        tls = 0
        for i in range(0, len(x_test), 256):
            x = x_test[i:i+256]
            t = t_test[i:i+256]
            y = mynet.forward(x)
            tls += np.sum(loss.loss(y, np.eye(10)[t]))
        nloss = tls/len(x_test)
        print(f"training loop: {j}, avg loss: {nloss}")
        if nloss < mloss:
            mloss = nloss
        else:
            if stop_time < 10:
                stop_time += 1
                continue
            else:
                break
