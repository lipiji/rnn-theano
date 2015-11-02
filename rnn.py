#pylint: skip-file
import time
import numpy as np
import theano
import theano.tensor as T
from softmax_layer import *
from gru_layer import *
from updates import *

class RNN(object):
    def __init__(self, in_size, out_size, hidden_size):
        X = T.matrix("X")
        self.create_model(X, in_size, out_size, hidden_size)

    def create_model(self, X, in_size, out_size, hidden_size):
        self.layers = []
        self.X = X
        self.n_hlayers = len(hidden_size)
        self.params = []

        for i in xrange(self.n_hlayers):
            if i == 0:
                layer_input = X
                shape = (in_size, hidden_size[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (hidden_size[i - 1], hidden_size[i])

            hidden_layer = GRULayer(shape, layer_input)
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        output_layer = SoftmaxLayer((hidden_layer.out_size, out_size), hidden_layer.activation)
        self.layers.append(output_layer)

        self.params += output_layer.params

        self.create_funs(X)
    
    def create_funs(self, X):
        activation = self.layers[len(self.layers) - 1].activation
        Y = T.matrix("Y")
        cost = T.mean(T.nnet.categorical_crossentropy(activation, Y))
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # try: 
        #updates = sgd(self.params, gparams, lr)
        #updates = momentum(self.params, gparams, lr)
        updates = rmsprop(self.params, gparams, lr)
        #updates = adagrad(self.params, gparams, lr)
        #updates = dadelta(self.params, gparams, lr)
        #updates = adam(self.params, gparams, lr)

        self.train = theano.function(inputs = [X, Y, lr], outputs = [cost], updates = updates)
        self.predict = theano.function(inputs = [X], outputs = [activation])
    
    def train(self, X, Y, lr):
        return self.train(X, Y, lr)

    def predict(self, X):
        return self.predict(X)
