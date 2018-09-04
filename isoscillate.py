import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import PIL
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('./result_ln6', exist_ok=True)

# Load the MNIST dataset
train, test = chainer.datasets.get_mnist()
xs, ts = train._datasets
txs, tts = test._datasets

xs_9 = []
ts_9 = []
txs_9 = []
tts_9 = []

for i in range(ts.shape[0]):
    if ts[i] != 9:
        xs_9.append(xs[i])
        ts_9.append(ts[i])

for i in range(tts.shape[0]):
    if tts[i] != 9:
        txs_9.append(txs[i])
        tts_9.append(tts[i])

xs_9 = np.array(xs_9)
ts_9 = np.array(ts_9)
txs_9 = np.array(txs_9)
tts_9 = np.array(tts_9)

loop_n = 6


# Network definition
class SingleRNN(chainer.Chain):
    def __init__(self, input_shape, hidden1, classes):
        super(SingleRNN, self).__init__(
            l1=L.Linear(input_shape, hidden1, initial_bias=np.random.normal(0, 0.01, hidden1)),
            r1=L.Linear(hidden1, hidden1, initial_bias=np.random.normal(0, 0.01, hidden1)),
            l3=L.Linear(hidden1, classes, initial_bias=np.random.normal(0, 0.01, classes))
        )
        self.hidden1 = hidden1

    def reset_state(self):
        self.h1 = Variable(np.zeros((1, self.hidden1), dtype=np.float32))

    def __call__(self, x):
        self.h1 = F.tanh(self.r1(self.h1) + self.l1(x))
        for i in range(loop_n):
            self.h1 = F.tanh(self.r1(self.h1))
        y = self.l3(self.h1)
        return y


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(Variable(x))
        self.loss = F.softmax_cross_entropy(y, t)
        for i in range(1, x.shape[0]):
            y = self.predictor(x[i])
            self.loss += F.softmax_cross_entropy(y, t[i])
        return self.loss


for k in range(100):
    plt.figure()
    singlernn = SingleRNN(784, 200, 9)
    singlernn.reset_state()
    model = Classifier(singlernn)
    # Setup optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    for i in range(15000):
        index = np.random.choice(range(50000))
        # singlernn.reset_state()
        x = np.expand_dims(xs_9[index], axis=0)
        t = np.expand_dims(ts_9[index], axis=0)
        t = Variable(t)
        model.cleargrads()
        loss = model(x, t)
        loss.backward()
        optimizer.update()
        loss.unchain_backward()

    correct = 0
    wrong = 0
    for i in range(txs_9.shape[0]):
        singlernn.reset_state()
        x = np.expand_dims(txs_9[i], axis=0)
        y = F.softmax(singlernn(x))
        output = np.argmax(y.data[0])
        if output == tts_9[i]:
            correct = correct + 1
        else:
            wrong = wrong + 1
    print(k, correct / (correct + wrong))

    x = np.expand_dims(txs[12], axis=0)
    output = []
    singlernn.reset_state()
    gamma = 0
    for i in range(500):
        y = singlernn(x)
        output.append(y.data[0])
    output = np.array(output)
    for i in range(8):
        plt.plot(output.T[i][450:], label='{}'.format(int(i)))
    plt.legend()
    plt.savefig("./result_ln6/{}_ln6.png".format(k))
