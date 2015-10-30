#pylint: skip-file
import time
import numpy as np
import theano
import theano.tensor as T
from softmax_layer import *
from logistic_layer import *
from gru_layer import *
import utils_pg as Utils
import data

seqs, i2w, w2i = data.char_sequence()

lr = 0.2
hidden_size = [100, 100]
layers = []

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

for lay in xrange(len(hidden_size)):
    if lay == 0:
        shape = (dim_x, hidden_size[lay])
    else:
        shape = (hidden_size[lay - 1], hidden_size[lay])
    layers.append(GRULayer(shape))
layers.append(SoftmaxLayer((hidden_size[len(hidden_size) - 1], dim_y)))

nn = NN(layers)

#batch train
start = time.time()
for i in xrange(1000):
    acc = 0.0
    in_start = time.time()
    for s in xrange(len(seqs)):
        seq = seqs[s]
        X = seq[0 : len(seq) - 1, ] 
        Y = seq[1 : len(seq), ]
        nn.batch_train(X, Y, lr)
    in_time = time.time() - in_start

    num_x = 0.0
    for s in xrange(len(seqs)):
        seq = seqs[s]
        X = seq[0 : len(seq) - 1, ] 
        Y = seq[1 : len(seq), ]

        label = np.argmax(Y, axis=1)
        p_label = np.argmax(nn.predict(X), axis=1)
        for c in xrange(len(label)):
            num_x += 1
            if label[c] == p_label[c]:
                acc += 1

    print i, acc / num_x, in_time
print time.time() - start
