import sys, os
#sys.path.append(os.pardir)#親ディレクトリのパスを通す
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
from common.functions import sigmoid, softmax
import pickle

#xは値,tはラベル#ラベルには文字がそのインデックスである確率が入っている
#0~1の値に正規化して、ラベルは最大で1その他は0
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#訓練データは60000個 x 784px
print(x_train.shape)
print(t_train.shape)


# === ミニバッチ処理 === # 
#莫大なデータの中から無造作ある枚数だけ選び出して、学習を行う

train_size = x_train.shape[0]
batch_size = 10
#0~60000までの乱数で埋めたサイズがbatch_sizeの配列
batch_mask = np.random.choice(train_size, batch_size)
# print(train_size)
# print(batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(t_batch)

