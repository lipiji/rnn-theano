#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class GRULayer(object):
    def __init__(self, rng, shape, X, is_train = 1, p = 0.5):
        self.in_size, self.out_size = shape
        
        self.W_xr = init_weights((self.in_size, self.out_size))
        self.W_hr = init_weights((self.out_size, self.out_size))
        self.b_r = init_bias(self.out_size)
        
        self.W_xz = init_weights((self.in_size, self.out_size))
        self.W_hz = init_weights((self.out_size, self.out_size))
        self.b_z = init_bias(self.out_size)

        self.W_xh = init_weights((self.in_size, self.out_size))
        self.W_hh = init_weights((self.out_size, self.out_size))
        self.b_h = init_bias(self.out_size)

        self.X = X

        def _active(x, pre_h):
            r = T.nnet.sigmoid(T.dot(x, self.W_xr) + T.dot(pre_h, self.W_hr) + self.b_r)
            z = T.nnet.sigmoid(T.dot(x, self.W_xz) + T.dot(pre_h, self.W_hz) + self.b_z)
            gh = T.tanh(T.dot(x, self.W_xh) + T.dot(r * pre_h, self.W_hh) + self.b_h)
            h = z * pre_h + (1 - z) * gh
            return h
        h, updates = theano.scan(_active, sequences = [self.X],
                                 outputs_info = [T.alloc(floatX(0.), 1, self.out_size)])
        #outputs_info = [dict(initial = T.zeros([1, self.out_size], dtype = self.X.dtype))]
       
        h = T.reshape(h, (self.X.shape[0], self.out_size))
        # dropout
        if p > 0:
            srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
            mask = srng.binomial(n = 1, p = 1-p, size = h.shape, dtype = theano.config.floatX)
            self.activation = T.switch(T.eq(is_train, 1), h * mask, h * (1 - p))
        else:
            self.activation = T.switch(T.eq(is_train, 1), h, h)
       
        self.params = [self.W_xr, self.W_hr, self.b_r, \
                       self.W_xz, self.W_hz, self.b_z, \
                       self.W_xh, self.W_hh, self.b_h]
