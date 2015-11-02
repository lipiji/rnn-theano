#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class LogisticLayer(object):
    def __init__(self, shape, X):
        self.in_size, self.out_size = shape

        self.W = init_weights(shape)
        self.b = init_bias(self.out_size)

        self.X = X

        self.activation = T.nnet.sigmoid(T.dot(self.X, self.W) + self.b)
        
        self.params = [self.W, self.b]
