#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class SoftmaxLayer(object):
    def __init__(self, shape, X, batch_size = 1):
        prefix = "Softmax_"
        self.in_size, self.out_size = shape
        self.W = init_weights(shape, prefix + "W", sample = "xavier")
        self.b = init_bias(self.out_size, prefix + "b")
        self.params = [self.W, self.b]
      
        a = T.dot(X, self.W) + self.b
        a_shape = a.shape
        a = T.nnet.softmax(T.reshape(a, (a_shape[0] * a_shape[1], a_shape[2])))
        self.activation = T.reshape(a, a_shape)

