import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
from dataset.paru_shushu.load_parushushu import load_parushushu

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# network = TwoLayerNet(input_size=30000, hidden_size= 50,output_size = 2)

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 0~train_sizeまでのランダムな値で埋めた大きさbatch_sizeの配列を取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 損失の変異を記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(i, ":", loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

plt.plot([x for x in range(len(train_loss_list))], train_loss_list)
plt.show()
