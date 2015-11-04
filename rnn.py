#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

from softmax import *
from gru import *
from lstm import *
from logistic import *
from updates import *

class RNN(object):
    def __init__(self, in_size, out_size, hidden_size, cell = "gru", p = 0.5):
        X = T.matrix("X")
        self.n_hlayers = len(hidden_size)
        self.layers = []
        self.params = []
        self.is_train = T.iscalar('is_train') # for dropout
        self.batch_size = T.iscalar('batch_size') # for mini-batch training
        
        rng = np.random.RandomState(1234)

        for i in xrange(self.n_hlayers):
            if i == 0:
                layer_input = X
                shape = (in_size, hidden_size[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (hidden_size[i - 1], hidden_size[i])

            if cell == "gru":
                hidden_layer = GRULayer(rng, str(i), shape, layer_input, self.is_train, self.batch_size, p)
            elif cell == "lstm":
                hidden_layer = LSTMLayer(rng, str(i), shape, layer_input, self.is_train, self.batch_size, p)
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        #hidden_layer = LogisticLayer(str(i + 1), (hidden_layer.out_size, hidden_layer.out_size), hidden_layer.activation)
        #self.layers.append(hidden_layer)
        #self.params += hidden_layer.params

        output_layer = SoftmaxLayer((hidden_layer.out_size, out_size), hidden_layer.activation, self.batch_size)
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
        
        self.train = theano.function(inputs = [X, Y, lr, self.batch_size], givens={self.is_train : np.cast['int32'](1)}, outputs = [cost], updates = updates)
        self.predict = theano.function(inputs = [X, self.batch_size], givens={self.is_train : np.cast['int32'](0)}, outputs = [activation])
    
