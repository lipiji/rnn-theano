#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class LSTMLayer(object):
    def __init__(self, rng, layer_id, shape, X, is_train = 1, p = 0.5):
        prefix = "LSTM_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape
        
        self.W_xi = init_weights((self.in_size, self.out_size), prefix + "W_xi" + layer_id)
        self.W_hi = init_weights((self.out_size, self.out_size), prefix + "W_hi" + layer_id)
        self.W_ci = init_weights((self.out_size, self.out_size), prefix + "W_ci" + layer_id)
        self.b_i = init_bias(self.out_size, prefix + "b_i" + layer_id)
        
        self.W_xf = init_weights((self.in_size, self.out_size), prefix + "W_xf" + layer_id)
        self.W_hf = init_weights((self.out_size, self.out_size), prefix + "W_hf" + layer_id)
        self.W_cf = init_weights((self.out_size, self.out_size), prefix + "W_cf" + layer_id)
        self.b_f = init_bias(self.out_size, prefix + "b_f" + layer_id)

        self.W_xc = init_weights((self.in_size, self.out_size), prefix + "W_xc" + layer_id)
        self.W_hc = init_weights((self.out_size, self.out_size), prefix + "W_hc" + layer_id)
        self.b_c = init_bias(self.out_size, prefix + "b_c" + layer_id)

        self.W_xo = init_weights((self.in_size, self.out_size), prefix + "W_xo" + layer_id)
        self.W_ho = init_weights((self.out_size, self.out_size), prefix + "W_ho" + layer_id)
        self.W_co = init_weights((self.out_size, self.out_size), prefix + "W_co" + layer_id)
        self.b_o = init_bias(self.out_size, prefix + "b_o" + layer_id)

        self.X = X
        
        def _active(x, pre_h, pre_c):
            i = T.nnet.sigmoid(T.dot(x, self.W_xi) + T.dot(pre_h, self.W_hi) + T.dot(pre_c, self.W_ci) + self.b_i)
            f = T.nnet.sigmoid(T.dot(x, self.W_xf) + T.dot(pre_h, self.W_hf) + T.dot(pre_c, self.W_cf) + self.b_f)
            gc = T.tanh(T.dot(x, self.W_xc) + T.dot(pre_h, self.W_hc) + self.b_c)
            c = f * pre_c + i * gc
            o = T.nnet.sigmoid(T.dot(x, self.W_xo) + T.dot(pre_h, self.W_ho) + T.dot(c, self.W_co) + self.b_o)
            h = o * T.tanh(c)
            return h, c
        [h, c], updates = theano.scan(_active,
                                      sequences = [self.X],
                                      outputs_info = [T.alloc(floatX(0.), 1, self.out_size),
                                                      T.alloc(floatX(0.), 1, self.out_size)])
        
        h = T.reshape(h, (self.X.shape[0], self.out_size))
        # dropout
        if p > 0:
            srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
            mask = srng.binomial(n = 1, p = 1-p, size = h.shape, dtype = theano.config.floatX)
            self.activation = T.switch(T.eq(is_train, 1), h * mask, h * (1 - p))
        else:
            self.activation = T.switch(T.eq(is_train, 1), h, h)
        
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                       self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc,            self.b_c,
                       self.W_xo, self.W_ho, self.W_co, self.b_o]
