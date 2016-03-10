#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *
import data

use_gpu(0)

e = 0.01
lr = 0.8
drop_rate = 0.
batch_size = 20
hidden_size = [100, 100]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 

seqs, i2w, w2i, data_xy = data.char_sequence("/data/toy.txt", batch_size)
dim_x = len(w2i)
dim_y = len(w2i)
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate)

print "training..."
start = time.time()
g_error = 9999.9999
for i in xrange(200):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
        X = xy[0] 
        Y = xy[1]
        maskX = xy[2]
        maskY = xy[3]
        local_batch_size = xy[4]
        cost = model.train(X, maskX, Y, maskY, lr, local_batch_size)
        error += cost
    in_time = time.time() - in_start

    error /= len(seqs);
    if error < g_error:
        g_error = error
        save_model("./model/rnn.model_" + str(i), model)

    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/char_rnn.model", model)

