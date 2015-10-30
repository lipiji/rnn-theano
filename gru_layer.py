#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from nn import *

class GRU(object):
    def __init__(self, shape):
        self.in_size, self.out_size = shape
        
        self.W_xr = init_weights((self.in_size, self.out_size))
        self.W_hr = init_weights((self.out_size, self.out_size))
        self.b_r = init_bias(self.out_size)
        
        self.W_xz = init_weights((self.in_size, self.out_size))
        self.W_hz = init_weights((self.out_size, self.out_size))
        self.b_z = init_bias(self.out_size)

        self.W_xh = init_weights((self.in_size, self.out_size))
        self.W_hh = init_weights((self.out_size, self.out_size))
        self.b_h = init_bias(self.out_size)

        # for gradients
        self.gW_xr = init_gradws((self.in_size, self.out_size))
        self.gW_hr = init_gradws((self.out_size, self.out_size))
        self.gb_r = init_bias(self.out_size)
        
        self.gW_xz = init_gradws((self.in_size, self.out_size))
        self.gW_hz = init_gradws((self.out_size, self.out_size))
        self.gb_z = init_bias(self.out_size)

        self.gW_xh = init_gradws((self.in_size, self.out_size))
        self.gW_hh = init_gradws((self.out_size, self.out_size))
        self.gb_h = init_bias(self.out_size)
    
        def _active(x, pre_h):
            r = T.nnet.sigmoid(T.dot(x, self.W_xr) + T.dot(pre_h, self.W_hr) + self.b_r)
            z = T.nnet.sigmoid(T.dot(x, self.W_xz) + T.dot(pre_h, self.W_hz) + self.b_z)
            gh = T.tanh(T.dot(x, self.W_xh) + T.dot(r * pre_h, self.W_hh) + self.b_h)
            h = z * pre_h + (1 - z) * gh
            return r, z, gh, h 
        X = T.matrix("X")
        H = T.matrix("H")
        [r, z, gh, h], updates = theano.scan(_active, sequences=[X], outputs_info=[None, None, None, H])
        self.active = theano.function(
            inputs = [X, H],
            outputs = [r, z, gh, h]
        )

        # TODO ->scan
        def _derive(prop, r, post_r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz):
            dh = prop \
                + T.dot(post_dr, self.W_hr.T) \
                + T.dot(post_dz, self.W_hz.T) \
                + T.dot(post_dgh * post_r, self.W_hh.T) \
                + post_dh * z
            dgh = dh * (1 - z) * (1 - gh ** 2)
            dr = T.dot(dgh * pre_h, self.W_hh.T) * ((1 - r) * r)
            dz = (1 - dh * (gh - pre_h)) * ((1 - z) * z)
            return dh, dgh, dr, dz
        prop, r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz, post_r = \
                T.matrices("prop", "r", "z", "gh", "pre_h", "post_dh", "post_dgh", "post_dr", "post_dz", "post_r")
        self.derive = theano.function(
            inputs = [prop, r, post_r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz],
            outputs = _derive(prop, r, post_r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz)
        )

        x, dz, dr, dgh = T.rows("x", "dz", "dr", "dgh")
        updates_grad = [(self.gW_xr, self.gW_xr + T.dot(x.T, dr)),
               (self.gW_xz, self.gW_xz + T.dot(x.T, dz)),
               (self.gW_xh, self.gW_xh + T.dot(x.T, dgh)),
               (self.gW_hr, self.gW_hr + T.dot(pre_h.T, dr)),
               (self.gW_hz, self.gW_hz + T.dot(pre_h.T, dz)),
               (self.gW_hh, self.gW_hh + T.dot((r * pre_h).T, dgh)),
               (self.gb_r, self.gb_r + dr),
               (self.gb_z, self.gb_z + dz),
               (self.gb_h, self.gb_h + dgh)]
        self.grad = theano.function(
            inputs = [x, r, pre_h, dz, dr, dgh],
            updates = updates_grad
        )

        updates_clear = [
               (self.gW_xr, self.gW_xr * 0),
               (self.gW_xz, self.gW_xz * 0),
               (self.gW_xh, self.gW_xh * 0),
               (self.gW_hr, self.gW_hr * 0),
               (self.gW_hz, self.gW_hz * 0),
               (self.gW_hh, self.gW_hh * 0),
               (self.gb_r, self.gb_r * 0),
               (self.gb_z, self.gb_z * 0),
               (self.gb_h, self.gb_h * 0)]
        self.clear_grad = theano.function(
            inputs = [],
            updates = updates_clear
        )

        lr = T.scalar()
        t = T.scalar()
        tm1 = T.scalar()
        updates_w = [
               (self.W_xr, self.W_xr - self.gW_xr * lr / t),
               (self.W_xz, self.W_xz - self.gW_xz * lr / t),
               (self.W_xh, self.W_xh - self.gW_xh * lr / t),
               (self.W_hr, self.W_hr - self.gW_hr * lr / tm1),
               (self.W_hz, self.W_hz - self.gW_hz * lr / tm1),
               (self.W_hh, self.W_hh - self.gW_hh * lr / tm1),
               (self.b_r, self.b_r - self.gb_r * lr / t),
               (self.b_z, self.b_z - self.gb_z * lr / t),
               (self.b_h, self.b_h - self.gb_h * lr / t)]
        self.update = theano.function(
            inputs = [lr, t, tm1],
            updates = updates_w
        )

        DZ, DR, DGH = T.matrices("DZ", "DR", "DGH")
        def _propagate(DR, DZ, DGH):
            return (T.dot(DR, self.W_xr.T) + T.dot(DZ, self.W_xz.T) + T.dot(DGH, self.W_xh.T))
        self.propagate = theano.function(inputs = [DR, DZ, DGH], outputs = _propagate(DR, DZ, DGH))


class GRULayer(Layer):
    def __init__(self, shape):
        self.h_size = shape[1]
        self.cell = GRU(shape)

        self.activation = []
        self.propagation = []

        self.R = []
        self.Z = []
        self.GH = []
        self.DH = []
        self.DGH = []
        self.DR = []
        self.DZ = []

    def active(self, X):
        pre_h = np.zeros((1, self.h_size), dtype=theano.config.floatX);
        [R, Z, GH, H] = self.cell.active(X, pre_h)
        self.activation = np.asmatrix(H)
        self.R = np.asmatrix(R)
        self.Z = np.asmatrix(Z)
        self.GH = np.asmatrix(GH)

    def calculate_delta(self, propagation = None, Y = None):
        DY = propagation
        DH = np.zeros((DY.shape), dtype=theano.config.floatX)
        DGH = np.copy(DH)
        DR = np.copy(DH)
        DZ = np.copy(DH)
        for t in xrange(DY.shape[0] - 1, -1, -1):
            pre_h = self.get_pre_h(t, self.h_size, self.activation)
            
            if t == (DY.shape[0] - 1):
                post_dh = np.zeros((1, self.h_size), dtype=theano.config.floatX)
                post_dgh = np.copy(post_dh)
                post_dr = np.copy(post_dh)
                post_dz = np.copy(post_dh)
                post_r = np.copy(post_dh)
            else:
                post_dh = DH[t + 1,]
                post_dgh = DGH[t + 1,]
                post_dr = DR[t + 1,]
                post_dz = DZ[t + 1,]
                post_r = self.R[t + 1,]
            
            dh, dgh, dr, dz = self.cell.derive(DY[t,], self.R[t,], post_r, self.Z[t,], self.GH[t,],
                                              pre_h, np.asmatrix(post_dh), np.asmatrix(post_dgh),
                                              np.asmatrix(post_dr), np.asmatrix(post_dz))
            DH[t,] = dh
            DGH[t,] = dgh
            DR[t,] = dr
            DZ[t,] = dz
        self.DH = np.asmatrix(DH)
        self.DGH = np.asmatrix(DGH)
        self.DR = np.asmatrix(DR)
        self.DZ = np.asmatrix(DZ)
        self.propagation = np.asmatrix(self.cell.propagate(self.DR, self.DZ, self.DGH))

    def update(self, X, lr):
        self.cell.clear_grad()
        for t in xrange(len(X)):
            pre_h = self.get_pre_h(t, self.h_size, self.activation)
            self.cell.grad(X[t,], self.R[t,], pre_h,  self.DZ[t,], self.DR[t,], self.DGH[t,])

        t = len(X)
        tm1 = t - 1
        if tm1 < 1:
            tm1 = 1
        self.cell.update(lr, t, tm1);


    def get_pre_h(self, t, size, H):
        if t == 0:
            return np.zeros((1, size), dtype=theano.config.floatX)
        else:
            return H[t,]

