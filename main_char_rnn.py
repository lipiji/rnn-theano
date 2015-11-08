#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *

use_gpu(0)

import data
drop_rate = 0.
batch_size = 1000
seqs, i2w, w2i, data_xy = data.char_sequence("/data/shakespeare.txt", batch_size)
hidden_size = [500, 400, 400]
dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

cell = "gru" # cell = "gru" or "lstm"
optimizer = "adadelta"

print "building..."
model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate)
print "load model..."
model = load_model("./model/char_rnn.model", model)

num_x = 0.0
acc = 0.0
for s in xrange(len(seqs)):
    seq = seqs[s]
    X = seq[0 : len(seq) - 1, ] 
    Y = seq[1 : len(seq), ]
    label = np.argmax(Y, axis=1)
    p_label = np.argmax(model.predict(X, np.ones((X.shape[0], 1), np.float32), 1)[0], axis=1)

    print i2w[np.argmax(X[0,])], 
    for c in xrange(len(label)):
        num_x += 1
        if label[c] == p_label[c]:
            acc += 1
        print i2w[p_label[c]],
    print "\n",
print "Accuracy = " + str(acc / num_x)

X = np.zeros((1, dim_x), np.float32)
a = "a"
X[0, w2i[a]] = 1
print a,
for i in xrange(100):
    Y = model.predict(X, np.ones((X.shape[0], 1), np.float32),  1)[0]
    Y = Y[Y.shape[0] - 1,:]
    p_label = np.argmax(Y)
    print i2w[p_label],
    X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)

