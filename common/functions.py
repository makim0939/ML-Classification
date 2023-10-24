import numpy as np

#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

#ソフトマックス関数
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

#交差エントロピー誤差（ミニバッチ処理対応版）
def cross_entropy_error(y:np.ndarray, t:np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]    
    delta = 1e-7
    #このバッチの交差エントロピー誤差の平均で近似
    return -np.sum(t * np.log(y+delta)) / batch_size
    #return -np.sum(np.log(y[np.arange(batch_size), t]+delta)) / batch_size

#数値微分(中心差分)
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

#勾配の算出
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        
        tmp = x[i] #tmpは元のxの値を保持
        #f(x+h)を求める
        x[i] = tmp + h
        fx1 = f(x)
        #f(x-1)を求める
        x[i] = tmp - h
        fx2 = f(x)
        
        x[i] = tmp
        grad[i] = (fx1-fx2) / (2*h)
        
    return grad

        
        
#勾配降下法 (lrはlerning rate　学習率)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        #lr*勾配の方向に下る。 += にすると勾配上昇法になる
        x -= lr*grad
    
    return x
    
        
        