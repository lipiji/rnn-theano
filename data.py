#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def char_sequence():
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    #f = open(curr_path + "/shakespeare.txt", "r")
    f = open(curr_path + "/data/toy.txt", "r")
    for line in f:
        line = line.strip('\n')
        if len(line) < 3:
            continue
        lines.append(line)
        for char in line:
            if char not in w2i:
                i2w[len(w2i)] = char
                w2i[char] = len(w2i)
    f.close()

    for i in range(0, len(lines)):
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(np.asmatrix(x))
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i


#data: http://deeplearning.net/data/mnist/mnist.pkl.gz
def mnist(batch_size = 1):
    def batch(X, Y, batch_size):
        data_xy = {}
        batch_x = []
        batch_y = []
        batch_id = 0
        for i in xrange(len(X)):
            batch_x.append(X[i])
            y = np.zeros((10), dtype = theano.config.floatX)
            y[Y[i]] = 1
            batch_y.append(y)
            if (len(batch_x) == batch_size) or (i == len(X) - 1):
                data_xy[batch_id] = [np.matrix(batch_x, dtype = theano.config.floatX), \
                                     np.matrix(batch_y, dtype = theano.config.floatX)]
                batch_id += 1
                batch_x = []
                batch_y = []
        return data_xy
    f = gzip.open(curr_path + "/data/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return batch(train_set[0], train_set[1], batch_size), \
           batch(valid_set[0], valid_set[1], batch_size), \
           batch(test_set[0], test_set[1], batch_size)

#data: http://deeplearning.net/data/mnist/mnist.pkl.gz
def shared_mnist():
    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        np_y = np.zeros((len(data_y), 10), dtype=theano.config.floatX)
        for i in xrange(len(data_y)):
            np_y[i, data_y[i]] = 1

        shared_x = theano.shared(np.asmatrix(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asmatrix(np_y, dtype=theano.config.floatX))
        return shared_x, shared_y
    f = gzip.open(curr_path + "/data/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]
