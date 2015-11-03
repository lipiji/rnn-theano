#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *

import data

seqs, i2w, w2i = data.char_sequence("/data/toy.txt")

e = 0.01
lr = 0.01
batch_size = 100
hidden_size = [100, 100]

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

cell = "gru"
print "building..."
model = RNN(dim_x, dim_y, hidden_size, cell, p = 0.) # cell = "gru" or "lstm"

print "training..."
start = time.time()
for i in xrange(100):
    error = 0.0;
    in_start = time.time()
    for s in xrange(len(seqs)):
        seq = seqs[s]
        X = seq[0 : len(seq) - 1, ] 
        Y = seq[1 : len(seq), ]
        error += model.train(X, Y, lr)[0]
    in_time = time.time() - in_start

    error /= len(seqs);
    if error <= e:
        break
   
    print "Iter = " + str(i) + ", Error = " + str(error / len(seqs)) + ", Time = " + str(in_time)
print time.time() - start

print "save model..."
save_model("rnn.model", model)

print "load model..."
loaded_model = RNN(dim_x, dim_y, hidden_size, cell)
loaded_model = load_model("rnn.model", loaded_model)

num_x = 0.0
acc = 0.0
for s in xrange(len(seqs)):
    seq = seqs[s]
    X = seq[0 : len(seq) - 1, ] 
    Y = seq[1 : len(seq), ]
    label = np.argmax(Y, axis=1)
    p_label = np.argmax(loaded_model.predict(X)[0], axis=1)

    print i2w[np.argmax(X[0,])], 
    for c in xrange(len(label)):
        num_x += 1
        if label[c] == p_label[c]:
            acc += 1
        print i2w[p_label[c]],
    print "\n",
print "Accuracy = " + str(acc / num_x)
