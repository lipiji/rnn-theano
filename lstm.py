#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from data import char_sequence

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_bias(size):
    return theano.shared(floatX(np.zeros((size, 1)))) 

class Cell(object):
    def __init__(self, shape):
        self.in_size, self.out_size = shape
        
        self.W_xi = init_weights((self.out_size, self.in_size))
        self.W_hi = init_weights((self.out_size, self.out_size))
        self.W_ci = init_weights((self.out_size, self.out_size))
        self.b_i = init_bias(self.out_size)
        
        self.W_xf = init_weights((self.out_size, self.in_size))
        self.W_hf = init_weights((self.out_size, self.out_size))
        self.W_cf = init_weights((self.out_size, self.out_size))
        self.b_f = init_bias(self.out_size)

        self.W_xc = init_weights((self.out_size, self.in_size))
        self.W_hc = init_weights((self.out_size, self.out_size))
        self.b_c = init_bias(self.out_size)

        self.W_xo = init_weights((self.out_size, self.in_size))
        self.W_ho = init_weights((self.out_size, self.out_size))
        self.W_co = init_weights((self.out_size, self.out_size))
        self.b_o = init_bias(self.out_size)
