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
    def __init__(self, in_size, out_size, hidden_size,
                 cell = "gru", optimizer = "rmsprop", p = 0.5):
        self.X = T.matrix("X")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.drop_rate = p
        self.is_train = T.iscalar('is_train') # for dropout
        self.batch_size = T.iscalar('batch_size') # for mini-batch training
        self.mask = T.matrix("mask")
        self.optimizer = optimizer
        self.define_layers()
        self.define_train_test_funcs()
        
    def define_layers(self):
        self.layers = []
        self.params = []
        rng = np.random.RandomState(1234)
        # hidden layers
        for i in xrange(len(self.hidden_size)):
            if i == 0:
                layer_input = self.X
                shape = (self.in_size, self.hidden_size[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (self.hidden_size[i - 1], self.hidden_size[i])

            if self.cell == "gru":
                hidden_layer = GRULayer(rng, str(i), shape, layer_input,
                                        self.mask, self.is_train, self.batch_size, self.drop_rate)
            elif self.cell == "lstm":
                hidden_layer = LSTMLayer(rng, str(i), shape, layer_input,
                                         self.mask, self.is_train, self.batch_size, self.drop_rate)
            
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        # output layer
        output_layer = SoftmaxLayer((hidden_layer.out_size, self.out_size),
                                    hidden_layer.activation, self.batch_size)
        self.layers.append(output_layer)
        self.params += output_layer.params
   
    # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15
    def categorical_crossentropy(self, y_pred, y_true):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

    def define_train_test_funcs(self):
        activation = self.layers[len(self.layers) - 1].activation
        self.Y = T.matrix("Y")
        cost = self.categorical_crossentropy(activation, self.Y)
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)

        #updates = sgd(self.params, gparams, lr)
        #updates = momentum(self.params, gparams, lr)
        #updates = rmsprop(self.params, gparams, lr)
        #updates = adagrad(self.params, gparams, lr)
        #updates = dadelta(self.params, gparams, lr)
        #updates = adam(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.X, self.mask, self.Y, lr, self.batch_size],
                                               givens = {self.is_train : np.cast['int32'](1)},
                                               outputs = [cost],
                                               updates = updates)
        self.predict = theano.function(inputs = [self.X, self.mask, self.batch_size],
                                                 givens = {self.is_train : np.cast['int32'](0)},
                                                 outputs = [activation])
  
        #theano.printing.pydotprint(self.train, outfile="./model/train.png", var_with_name_simple=True) 
