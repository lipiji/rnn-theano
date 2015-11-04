#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class SoftmaxLayer(object):
    def __init__(self, shape, X, batch_size = 1):
        prefix = "Softmax_"
        self.in_size, self.out_size = shape
        self.W = init_weights(shape, prefix + "W")
        self.b = init_bias(self.out_size, prefix + "b")
        self.X = X
        self.params = [self.W, self.b]
        
        def _active(x):
            x = T.reshape(x, (batch_size, self.in_size))
            o = T.nnet.softmax(T.dot(x, self.W) + self.b)
            o = T.reshape(o, (1, batch_size * self.out_size))
            return o
        o, updates = theano.scan(_active, sequences = [self.X])
        self.activation = T.reshape(o, (self.X.shape[0], batch_size * self.out_size))



