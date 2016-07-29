#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class LSTMLayer(object):
    def __init__(self, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        prefix = "LSTM_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape
        
        self.W_x_ifoc = init_weights_4((self.in_size, self.out_size), prefix + "W_x_ifoc" + layer_id, sample = "xavier")
        self.W_h_ifoc = init_weights_4((self.out_size, self.out_size), prefix + "W_h_ifoc" + layer_id, sample = "ortho")
        self.W_c_if = init_weights_2((self.out_size, self.out_size), prefix + "W_c_if" + layer_id, sample = "ortho")
        self.W_c_o = init_weights((self.out_size, self.out_size), prefix + "W_c_o" + layer_id, sample = "ortho")
        self.b_ifoc = init_bias(self.out_size * 4, prefix + "b_ifoc" + layer_id)

        self.params = [self.W_x_ifoc, self.W_h_ifoc, self.W_c_if, self.W_c_o, self.b_ifoc]

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim : (n + 1) * dim]
            return _x[:, n * dim : (n + 1) * dim]

        X_4ifoc = T.dot(X, self.W_x_ifoc) + self.b_ifoc
        def _active(m, x_4ifoc, pre_h, pre_c, W_h_ifoc, W_c_if, W_c_o):
            ifoc_preact = x_4ifoc + T.dot(pre_h, W_h_ifoc)
            c_if_preact = T.dot(pre_c, W_c_if)

            i = T.nnet.sigmoid(_slice(ifoc_preact, 0, self.out_size) + _slice(c_if_preact, 0, self.out_size))
            f = T.nnet.sigmoid(_slice(ifoc_preact, 1, self.out_size) + _slice(c_if_preact, 1, self.out_size))
            gc = T.tanh(_slice(ifoc_preact, 2, self.out_size))
            c = f * pre_c + i * gc
            o = T.nnet.sigmoid(_slice(ifoc_preact, 3, self.out_size) + T.dot(c, W_c_o))
            h = o * T.tanh(c)

            c = c * m[:, None]
            h = h * m[:, None]
            return h, c
        [h, c], updates = theano.scan(_active,
                                      sequences = [mask, X_4ifoc],
                                      outputs_info = [T.alloc(floatX(0.), batch_size, self.out_size),
                                                      T.alloc(floatX(0.), batch_size, self.out_size)],
                                      non_sequences = [self.W_h_ifoc, self.W_c_if, self.W_c_o],
                                      strict = True)
        
        # dropout
        if p > 0:
            srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
            drop_mask = srng.binomial(n = 1, p = 1-p, size = h.shape, dtype = theano.config.floatX)
            self.activation = T.switch(T.eq(is_train, 1), h * drop_mask, h * (1 - p))
        else:
            self.activation = T.switch(T.eq(is_train, 1), h, h)
        
        

class BdLSTM(object):
    # Bidirectional LSTM Layer.
    def __init__(self, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        fwd = LSTMLayer(rng, "_fwd_" + layer_id, shape, X, mask, is_train, batch_size, p)
        bwd = LSTMLayer(rng, "_bwd_" + layer_id, shape, X[::-1], mask[::-1], is_train, batch_size, p)
        self.params = fwd.params + bwd.params
        self.activation = T.concatenate([fwd.activation, bwd.activation[::-1]], axis=1)

