#pylint: skip-file
import time
import numpy as np
import theano
import theano.tensor as T
from softmax_layer import *
from logistic_layer import *
import utils_pg as Utils
import data

#seqs, i2w, w2i = data.char_sequence()

lr = 0.2
batch_size = 20
train_set, valid_set, test_set  = data.mnist(batch_size)

hidden_size = [500]
layers = []

#layers = [GRULayer(len(w2i), hidden_size[0]),
#          GRULayer(hidden_size[0], hidden_size[1]),
#          SoftmaxLayer(hidden_size[len(hidden_size) - 1], len(w2i))]

dim_x = train_set[0][0][0].shape[1]
dim_y = train_set[0][1][0].shape[1]
print dim_x, dim_y

for lay in xrange(len(hidden_size)):
    if lay == 0:
        shape = (dim_x, hidden_size[lay])
    else:
        shape = (hidden_size[lay - 1], hidden_size[lay])
    layers.append(LogisticLayer(shape))
layers.append(SoftmaxLayer((hidden_size[len(hidden_size) - 1], dim_y)))

nn = NN(layers)

#batch train
start = time.time()
for i in xrange(100):
    acc = 0.0
    in_start = time.time()
    for index, data_xy in train_set.items():
        X = data_xy[0]
        Y = data_xy[1]
        nn.batch_train(X, Y, lr)
    in_time = time.time() - in_start

    num_x = 0.0
    for index, data_xy in valid_set.items():
        X = data_xy[0]
        Y = data_xy[1]
        label = np.argmax(Y, axis=1)
        p_label = np.argmax(nn.predict(X), axis=1)
        for c in xrange(len(label)):
            num_x += 1
            if label[c] == p_label[c]:
                acc += 1

    print i, acc / num_x, in_time
print time.time() - start
