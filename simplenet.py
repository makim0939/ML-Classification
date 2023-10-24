import numpy as np
from common.functions import softmax, cross_entropy_error, gradient_descent

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def pridict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        #z=WX
        z = self.pridict(x)
        #活性化間数に通す
        y = softmax(z)
        #損失関数を求める
        loss = cross_entropy_error(y,t)
        