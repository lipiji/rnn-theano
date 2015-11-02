#pylint: skip-file
import time
import numpy as np
import theano
import theano.tensor as T
import utils_pg as Utils
from rnn import *
import data

seqs, i2w, w2i = data.char_sequence("/data/toy.txt")

lr = 0.01
batch_size = 100
hidden_size = [100, 100]

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

model = RNN(dim_x, dim_y, hidden_size, cell = "gru") # cell = "gru" or "lstm"

start = time.time()
for i in xrange(100):
    acc = 0.0
    in_start = time.time()
    for s in xrange(len(seqs)):
        seq = seqs[s]
        X = seq[0 : len(seq) - 1, ] 
        Y = seq[1 : len(seq), ]
        model.train(X, Y, lr)
    in_time = time.time() - in_start
    
    num_x = 0.0
    for s in xrange(len(seqs)):
        seq = seqs[s]
        X = seq[0 : len(seq) - 1, ] 
        Y = seq[1 : len(seq), ]
        label = np.argmax(Y, axis=1)
        p_label = np.argmax(model.predict(X)[0], axis=1)

        print i2w[np.argmax(X[0,])], 
        for c in xrange(len(label)):
            num_x += 1
            if label[c] == p_label[c]:
                acc += 1
            print i2w[p_label[c]],
        print "\n",
    print "Iter = " + str(i) + ", Accuracy = " + str(acc / num_x) + ", Time = " + str(in_time)
print time.time() - start
