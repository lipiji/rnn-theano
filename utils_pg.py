#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

def floatX(X):
    return X.astype(dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def init_gradws(shape):
    return theano.shared(floatX(np.zeros(shape)))

def init_bias(size):
    return theano.shared(floatX(np.zeros((1, size))), broadcastable=(True, False))

def rmse(py, y):
    e = 0
    for t in xrange(len(y)):
        e += np.sqrt(np.mean((np.asarray(py[t,]) - np.asarray(y[t,])) ** 2))
    return e / len(y)


