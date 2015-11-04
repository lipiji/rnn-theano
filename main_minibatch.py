#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *

import data

batch_size = 3
seqs, i2w, w2i, data_xy = data.char_sequence("/data/toy.txt", batch_size)

e = 0.01
lr = 0.01

hidden_size = [100, 100]

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

cell = "gru" # cell = "gru" or "lstm"

print "building..."
model = RNN(dim_x, dim_y, hidden_size, cell, p = 0.)

print "training..."
start = time.time()
g_error = 9999.9999
for i in xrange(100):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
        X = xy[0] 
        Y = xy[1]
        local_batch_size = xy[2]
        cost = model.train(X, Y, lr, local_batch_size)[0]
        error += cost
        #print i, g_error, s, "/", len(seqs), cost
    in_time = time.time() - in_start

    error /= len(seqs);
    if error < g_error:
        g_error = error
        save_model("./model/rnn.model_" + str(i), model)

    print "Iter = " + str(i) + ", Error = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/rnn.model", model)

print "load model..."
loaded_model = RNN(dim_x, dim_y, hidden_size, cell)
loaded_model = load_model("./model/rnn.model", loaded_model)

num_x = 0.0
acc = 0.0
for s in xrange(len(seqs)):
    seq = seqs[s]
    X = seq[0 : len(seq) - 1, ] 
    Y = seq[1 : len(seq), ]
    label = np.argmax(Y, axis=1)
    p_label = np.argmax(loaded_model.predict(X, 1)[0], axis=1)

    print i2w[np.argmax(X[0,])], 
    for c in xrange(len(label)):
        num_x += 1
        if label[c] == p_label[c]:
            acc += 1
        print i2w[p_label[c]],
    print "\n",
print "Accuracy = " + str(acc / num_x)
