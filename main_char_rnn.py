#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *

import data

batch_size = 1000
seqs, i2w, w2i, data_xy = data.char_sequence("/data/shakespeare.txt", batch_size)

print w2i 

hidden_size = [512, 512]
dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

cell = "gru" # cell = "gru" or "lstm"

print "building..."
model = RNN(dim_x, dim_y, hidden_size, cell, p = 0.5)

print "load model..."
model = load_model("./model/char_rnn.model", model)

X = np.zeros((1, dim_x), np.float32)
a = "a"
X[0, w2i[a]] = 1
print a,
for i in xrange(100):
    Y = model.predict(X, 1)[0]
    Y = Y[Y.shape[0] - 1,:]
    p_label = np.argmax(Y)
    print i2w[p_label],

    print X.shape
    print Y.shape
    X = np.concatenate((X, Y), axis=0)

