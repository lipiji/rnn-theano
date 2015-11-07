# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip
import jieba

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def char_sequence(f_path = None, batch_size = 1):
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    if f_path == None:
        f = open(curr_path + "/data/toy.txt", "r")
    else:
        f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
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

    data_xy = batch_sequences(seqs, i2w, w2i, batch_size)
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i, data_xy

def batch_sequences(seqs, i2w, w2i, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    zeros_m = np.zeros((1, len(w2i)), dtype = theano.config.floatX)
    for i in xrange(len(seqs)):
        seq = seqs[i];
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)

        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            max_len = np.max(seqs_len);
            mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
            
            concat_X = []
            concat_Y = []
            for b_i in xrange(len(batch_x)):
                X = batch_x[b_i]
                Y = batch_y[b_i]
                for r in xrange(max_len - X.shape[0]):
                    X = np.concatenate((X, zeros_m), axis=0)
                    Y = np.concatenate((Y, zeros_m), axis=0)
                if b_i == 0:
                    concat_X = X
                    concat_Y = Y
                else:
                    concat_X = np.concatenate((concat_X, X), axis=1)
                    concat_Y = np.concatenate((concat_Y, Y), axis=1)
                
                mask[0 : X.shape[0], b_i] = 1

            data_xy[batch_id] = [concat_X, concat_Y, mask, len(batch_x)]
            batch_x = []
            batch_y = []
            seqs_len = []
            batch_id += 1
    return data_xy

def load_hlm(f_path, batch_size = 1):
    jieba.load_userdict("./data/hlm/name.dic")
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    data_xy = {}
    f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        if len(line) < 3 or "手机电子书" in line:
            continue
        seg_list = jieba.cut(line)

        w_line = []
        for w in seg_list:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            w_line.append(w)
            if len(w_line) == 100:
                lines.append(w_line)
                w_line = []
        if len(w_line) < 100:
            lines.append(w_line)
    f.close
    seqs = lines
    data_xy = batch_index(seqs, i2w, w2i, batch_size)
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i, data_xy

# limit memory
def batch_index(seqs, i2w, w2i, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    for i in xrange(len(seqs)):
        batch_x.append(i)
        batch_y.append(i)
        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            data_xy[batch_id] = [batch_x, batch_y, [], len(batch_x)]
            batch_x = []
            batch_y = []
            batch_id += 1
    return data_xy

def index2seqs(lines, x_index, w2i):
    seqs = []
    for i in x_index:
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(np.asmatrix(x))

    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    zeros_m = np.zeros((1, len(w2i)), dtype = theano.config.floatX)
    for i in xrange(len(seqs)):
        seq = seqs[i];
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)

    max_len = np.max(seqs_len);
    mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
            
    concat_X = []
    concat_Y = []
    for b_i in xrange(len(batch_x)):
        X = batch_x[b_i]
        Y = batch_y[b_i]
        for r in xrange(max_len - X.shape[0]):
            X = np.concatenate((X, zeros_m), axis=0)
            Y = np.concatenate((Y, zeros_m), axis=0)
        if b_i == 0:
            concat_X = X
            concat_Y = Y
        else:
            concat_X = np.concatenate((concat_X, X), axis=1)
            concat_Y = np.concatenate((concat_Y, Y), axis=1)
                
        mask[0 : X.shape[0], b_i] = 1
    return concat_X, concat_Y, mask, len(batch_x)


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
