#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def init_gradws(shape):
    return theano.shared(floatX(np.zeros(shape)))

def init_bias(size):
    return theano.shared(floatX(np.zeros((1, size))), broadcastable=(True, False))


