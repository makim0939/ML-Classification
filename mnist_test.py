import sys
import os
# sys.path.append(os.pardir)#親ディレクトリのパスを通す
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
from common.functions import sigmoid, softmax
import pickle


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    # 取得するデータはピクセル値が0~1の、2次元配列
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# pickleファイルに保存されたデータを読み込む
# pickleは、実行中のオブジェクトをファイルに保存して、他の実行で呼び出せる

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        # ウェイト、バイアスの情報が入った辞書型オブジェクト
        network = pickle.load(f)

    return network

# 予めpickleのに保存されたウェイト、バイアスを用いて文字を推測


def pridict(network, x):
    # w1, w2, w3は各層のそれぞれのノードに対するウェイトが入っている配列
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

# 手作業で確認
IMG_INDEX = 1
x, t = get_data()
img = x
for i in range(len(img[0])):
    if img[IMG_INDEX][i] != 0:
        img[IMG_INDEX][i] = 255

img_show(img[IMG_INDEX].reshape(28, 28))
network = init_network()
y = pridict(network, x[IMG_INDEX])
result = np.argmax(y)
print(result)


# === 精度を確認 === #
network = init_network()
x, t = get_data()
count = 0
batch_size = 100
# print(range(0, len(x), batch_size)) = range(0, 10000, 100)
for i in range(0, len(x), batch_size):
    x_batch = x[i: i+batch_size]
    y_batch = pridict(network, x_batch)
    # axis = 1　は1次元目の要素ごとにargmaxを求めるという意味 y_batch[i]
    p = np.argmax(y_batch, axis=1)
    count += np.sum(p == t[i: i+batch_size])

acc = count/len(x)
print(x_batch[0].shape)
print(y_batch[0])
print(np.argmax(y_batch[0]))
print(acc)

# 出力層はソフトマックス関数
# ===== mnistから画像を読み込むのを確認 ===== #

# (訓練画像, 訓練ラベル), (テスト画像, テストラベル) = load_mnist
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# img = x_train[0]
# label = t_train[0]
# # print(label)
# img = img.reshape(28, 28)
# img_show(img)
# print("x_label", label)
