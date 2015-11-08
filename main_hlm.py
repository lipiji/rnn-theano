#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *
import data

# set use gpu programatically
use_gpu(1)

e = 0.01
lr = 0.2
drop_rate = 0.
batch_size = 50
hidden_size = [400, 400]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 

seqs, i2w, w2i, data_xy = data.load_hlm("/data/hlm/hlm.txt", batch_size)
dim_x = len(w2i)
dim_y = len(w2i)
print "#features = ", dim_x, "#labels = ", dim_y
print "compiling..."
model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate)

print "training..."
start = time.time()
g_error = 9999.9999
for i in xrange(100):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
        X, Y, mask, local_batch_size = data.index2seqs(seqs, xy[0], w2i)
        cost = model.train(X, mask, Y, lr, local_batch_size)[0]
        error += cost
        print i, g_error, batch_id, "/", len(data_xy), cost
    in_time = time.time() - in_start

    error /= len(seqs);
    if error < g_error:
        g_error = error
        save_model("./model/rnn_hlm.model_" + str(i), model)

    print "Iter = " + str(i) + ", Error = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/rnn_hlm.model", model)

print "load model..."
loaded_model = RNN(dim_x, dim_y, hidden_size, cell)
loaded_model = load_model("./model/rnn_hlm.model", loaded_model)

