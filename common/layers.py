
import numpy as np
from common.functions import softmax, cross_entropy_error
#ReLUレイヤー
class Relu:
    def __init__(self):
        self.mask = None

    #順伝播        
    def forward(self, x):
        #ReLU関数はx>0でout=xそれ以外で0
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    #逆伝播
    def backward(self, dout):
        #doutは上流から送られてきた微分
        dout[self.mask] = 0
        dx = dout
        
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None
    
   #順伝播        
    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        
        return out

    #逆伝播
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    #順伝播        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    #逆伝播 
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        #1バッチの合計
        self.db = np.sum(dout, axis=0)
        
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

        

#基礎的なレイヤー

#乗算レイヤー
class MulLayer:
    def __init__(self):
        #コンストラクタではxとyの宣言のみ
        sel
        f.x = None
        self.y = None

    #順伝播        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        
        return out
    
    #逆伝播（順伝播ののちに行わないと、x,yの値がNoneのまま）
    def backward(self, dout):
        #doutは上流から送られてきた微分
        #乗算ではz = x*y
        # dz/dx = y, dz/dy = x のようにxとyをひっくり返せば良い
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
    
    
#加算レイヤー
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None
    #順伝播  
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x+y
    
        return out
    #逆伝播
    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy
        
