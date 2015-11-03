#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class SoftmaxLayer(object):
    def __init__(self, shape, X):
        prefix = "Softmax_"
        self.in_size, self.out_size = shape
        self.W = init_weights(shape, prefix + "W")
        self.b = init_bias(self.out_size, prefix + "b")
        self.X = X
        self.activation = T.nnet.softmax(T.dot(X, self.W) + self.b)
        self.params = [self.W, self.b]

