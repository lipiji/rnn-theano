#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from nn import * 

class Logistic(object):
    def __init__(self, shape):
        self.in_size, self.out_size = shape

        self.W = init_weights(shape)
        self.b = init_bias(self.out_size)

        self.gW = init_gradws(shape)
        self.gb = init_bias(self.out_size)



        D, X = T.matrices("D", "X")
        def _active(X):
            return T.nnet.sigmoid(T.dot(X, self.W) + self.b)
        self.active = theano.function(inputs = [X], outputs = _active(X))
        
        def _derive(D, X):
            return D * ((1 - X) * X)
        self.derive = theano.function(
            inputs = [D, X],
            outputs = _derive(D, X)
        )

        def _propagate(D):
            return T.dot(D, self.W.T)
        self.propagate = theano.function(inputs = [D], outputs = _propagate(D))

        x, dy = T.rows("x","dy")
        updates_grad = [(self.gW, self.gW + T.dot(x.T, dy)),
               (self.gb, self.gb + dy)]
        self.grad = theano.function(
            inputs = [x, dy],
            updates = updates_grad
        )

        updates_clear = [
               (self.gW, self.gW * 0),
               (self.gb, self.gb * 0)]
        self.clear_grad = theano.function(
            inputs = [],
            updates = updates_clear
        )

        lr = T.scalar()
        t = T.scalar()
        updates_w = [
               (self.W, self.W - self.gW * lr / t),
               (self.b, self.b - self.gb * lr / t)]
        self.update = theano.function(
            inputs = [lr, t],
            updates = updates_w
        )

class LogisticLayer(Layer):
    def __init__(self, shape):
        self.cell = Logistic(shape)
        self.activation = []
        self.delta = []
        self.propagation = []

    def active(self, X):
        self.activation = np.asmatrix(self.cell.active(X))

    def calculate_delta(self, propagation = None, Y = None):
        self.delta = np.asmatrix(self.cell.derive(propagation, self.activation))
        self.propagation = np.asmatrix(self.cell.propagate(self.delta))

    def update(self, X, lr):
        self.cell.clear_grad()
        for t in xrange(len(X)):
            self.cell.grad(X[t,], self.delta[t,])
        self.cell.update(lr, len(X));

