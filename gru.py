#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class GRULayer(object):
    def __init__(self, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        prefix = "GRU_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape
        
        self.W_x_rz = init_weights_2((self.in_size, self.out_size), prefix + "W_x_rz" + layer_id, sample = "xavier")
        self.W_h_rz = init_weights_2((self.out_size, self.out_size), prefix + "W_h_rz" + layer_id, sample = "ortho")
        self.b_rz = init_bias(self.out_size * 2, prefix + "b_rz" + layer_id)
        
        self.W_xh = init_weights((self.in_size, self.out_size), prefix + "W_xh" + layer_id, sample = "xavier")
        self.W_hh = init_weights((self.out_size, self.out_size), prefix + "W_hh" + layer_id, sample = "ortho")
        self.b_h = init_bias(self.out_size, prefix + "b_h" + layer_id)

        X_4rz = T.dot(X, self.W_x_rz) + self.b_rz
        X_4h = T.dot(X, self.W_xh) + self.b_h

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim : (n + 1) * dim]
            return _x[:, n * dim : (n + 1) * dim]

        def _active(m, x_4rz, x_4h, pre_h, W_h_rz, W_hh):
            rz_preact = x_4rz + T.dot(pre_h, W_h_rz)
            r = T.nnet.sigmoid(_slice(rz_preact, 0, self.out_size))
            z = T.nnet.sigmoid(_slice(rz_preact, 1, self.out_size))
            gh = T.tanh(x_4h + T.dot(r * pre_h, W_hh))
            h = (1 - z) * pre_h + z * gh
            h = h * m[:, None]
            return h
        
        outputs, updates = theano.scan(_active,
                                       sequences = [mask, X_4rz, X_4h],
                                       outputs_info = [T.alloc(floatX(0.), batch_size, self.out_size)],
                                       non_sequences = [self.W_h_rz, self.W_hh],
                                       strict = True)
        h = outputs
        # dropout
        if p > 0:
            srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
            drop_mask = srng.binomial(n = 1, p = 1-p, size = h.shape, dtype = theano.config.floatX)
            self.activation = T.switch(T.eq(is_train, 1), h * drop_mask, h * (1 - p))
        else:
            self.activation = T.switch(T.eq(is_train, 1), h, h)
       
        self.params = [self.W_x_rz, self.W_h_rz, self.b_rz,
                       self.W_xh, self.W_hh, self.b_h]

class BdGRU(object):
    # Bidirectional GRU Layer.
    def __init__(self, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        fwd = GRULayer(rng, "_fwd_" + layer_id, shape, X, mask, is_train, batch_size, p)
        bwd = GRULayer(rng, "_bwd_" + layer_id, shape, X[::-1], mask[::-1], is_train, batch_size, p)
        self.params = fwd.params + bwd.params
        self.activation = T.concatenate([fwd.activation, bwd.activation[::-1]], axis=1)

