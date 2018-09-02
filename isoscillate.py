import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer  import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import PIL
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('./result', exist_ok=True)

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

loop_n = 8

# Network definition
class SingleRNN(chainer.Chain):
    def __init__(self, input_shape, hidden1, classes):
        super(SingleRNN, self).__init__(
            l1=L.Linear(input_shape, hidden1),
            r1=L.Linear(hidden1, hidden1),
            l3=L.Linear(hidden1, classes)
        )
        self.hidden1 = hidden1
        # self.hidden2 = hidden2

    def reset_state(self):
        self.h1 = Variable(np.zeros((1, self.hidden1), dtype=np.float32))
        # self.h2 = Variable(np.zeros((1,self.hidden2), dtype=np.float32))

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
    singlernn = SingleRNN(784, 100, 9)
    singlernn.reset_state()
    model = Classifier(singlernn)
    # Setup optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    for i in range(5000):
        index = np.random.choice(range(50000))
        # index = 0
        x = np.expand_dims(xs_9[index], axis=0)
        t = np.expand_dims(ts_9[index], axis=0)
        t = Variable(t)
        model.cleargrads()
        loss = model(x, t)
        if i % 1000 == 0:
            print(int(i / 1000), loss.data)
        loss.backward()
        optimizer.update()
        loss.unchain_backward()
        # loss.unchain_backward()

    x = np.expand_dims(txs[12], axis=0)
    output = []
    singlernn.reset_state()
    gamma = 0
    zero = np.zeros(784).astype("float32")
    zero = np.expand_dims(zero, axis=0)
    for i in range(500):
        y = singlernn(x)
        output.append(y.data[0])
        for j in range(8):
            y = singlernn(zero)
            y = singlernn(zero)
    output = np.array(output)
    for i in range(8):
        plt.plot(output.T[i][450:])
    plt.savefig("./result/{}.png".format(k))


