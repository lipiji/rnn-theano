#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class LogisticLayer(object):
    def __init__(self, layer_id, shape, X):
        prefix = "Logistic_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape
        self.W = init_weights(shape, prefix + "W" + layer_id)
        self.b = init_bias(self.out_size, prefix + "b" + layer_id)
        self.X = X
        self.activation = T.nnet.sigmoid(T.dot(self.X, self.W) + self.b)
        self.params = [self.W, self.b]
