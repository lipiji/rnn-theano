#pylint: skip-file
import time
import numpy as np
import theano
import theano.tensor as T
from data import char_sequence

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def init_gradws(shape):
    return theano.shared(floatX(np.zeros(shape)))

def init_bias(size):
    return theano.shared(floatX(np.zeros((size, 1)))) 

class Cell(object):
    def __init__(self, shape):
        self.in_size, self.out_size = shape
        
        self.W_xr = init_weights((self.out_size, self.in_size))
        self.W_hr = init_weights((self.out_size, self.out_size))
        self.b_r = init_bias(self.out_size)
        
        self.W_xz = init_weights((self.out_size, self.in_size))
        self.W_hz = init_weights((self.out_size, self.out_size))
        self.b_z = init_bias(self.out_size)

        self.W_xh = init_weights((self.out_size, self.in_size))
        self.W_hh = init_weights((self.out_size, self.out_size))
        self.b_h = init_bias(self.out_size)

        self.W_hy = init_weights((self.in_size, self.out_size))
        self.b_y = init_bias(self.in_size)
       
        # for gradients
        self.gW_xr = init_gradws((self.out_size, self.in_size))
        self.gW_hr = init_gradws((self.out_size, self.out_size))
        self.gb_r = init_bias(self.out_size)
        
        self.gW_xz = init_gradws((self.out_size, self.in_size))
        self.gW_hz = init_gradws((self.out_size, self.out_size))
        self.gb_z = init_bias(self.out_size)

        self.gW_xh = init_gradws((self.out_size, self.in_size))
        self.gW_hh = init_gradws((self.out_size, self.out_size))
        self.gb_h = init_bias(self.out_size)

        self.gW_hy = init_gradws((self.in_size, self.out_size))
        self.gb_y = init_bias(self.in_size)
    
        def _active(x, pre_h):
            r = T.nnet.sigmoid(T.dot(self.W_xr, x) + T.dot(self.W_hr, pre_h) + self.b_r)
            z = T.nnet.sigmoid(T.dot(self.W_xz, x) + T.dot(self.W_hz, pre_h) + self.b_z)
            gh = T.tanh(T.dot(self.W_xh, x) + T.dot(self.W_hh, r * pre_h) + self.b_h)
            h = z * pre_h + (1 - z) * gh
            return r, z, gh, h 
        x = T.matrix("x")
        pre_h = T.matrix("pre_h")
        self.active = theano.function(
            inputs = [x, pre_h],
            outputs = _active(x, pre_h)
        )
        
        def _predict(h):
            y = T.nnet.softmax((T.dot(self.W_hy, h) + self.b_y).T).T
            return y
        h = T.matrix("h")
        self.predict = theano.function(
            inputs = [h],
            outputs = _predict(h)
        )
        
        def _derive(y, py, r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz, post_r):
            dy = py - y;
            dh = T.dot(self.W_hy.T, dy) \
                + T.dot(self.W_hr.T, post_dr) \
                + T.dot(self.W_hz.T, post_dz) \
                + T.dot(self.W_hh.T, post_dgh * post_r) \
                + post_dh * z
            dgh = dh * (1 - z) * (1 - gh ** 2)
            dr = T.dot(self.W_hh.T, dgh * pre_h) * ((1 - r) * r)
            dz = (1 - dh * (gh - pre_h)) * ((1 - z) * z)
            return dy, dh, dgh, dr, dz
        y, py, r, z, gh, post_dh, post_dgh, post_dr, post_dz, post_r = \
            T.matrices("y", "py", "r", "z", "gh", "post_dh", "post_dgh", "post_dr", "post_dz", "post_r")
        self.derive = theano.function(
            inputs = [y, py, r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz, post_r],
            outputs = _derive(y, py, r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz, post_r)
        )

        dy, dz, dr, dgh = T.matrices("dy", "dz", "dr", "dgh")
        updates_grad = [(self.gW_xr, self.gW_xr + T.dot(dr, x.T)),
               (self.gW_xz, self.gW_xz + T.dot(dz, x.T)),
               (self.gW_xh, self.gW_xh + T.dot(dgh, x.T)),
               (self.gW_hr, self.gW_hr + T.dot(dr, pre_h.T)),
               (self.gW_hz, self.gW_hz + T.dot(dz, pre_h.T)),
               (self.gW_hh, self.gW_hh + T.dot(dgh, (r * pre_h).T)),
               (self.gW_hy, self.gW_hy + T.dot(dy, h.T)),
               (self.gb_r, self.gb_r + dr),
               (self.gb_z, self.gb_z + dz),
               (self.gb_h, self.gb_h + dgh),
               (self.gb_y, self.gb_y + dy)]
        self.grad = theano.function(
            inputs = [x, r, pre_h, h, dy, dz, dr, dgh],
            updates = updates_grad
        )

        updates_clear = [
               (self.gW_xr, self.gW_xr * 0),
               (self.gW_xz, self.gW_xz * 0),
               (self.gW_xh, self.gW_xh * 0),
               (self.gW_hr, self.gW_hr * 0),
               (self.gW_hz, self.gW_hz * 0),
               (self.gW_hh, self.gW_hh * 0),
               (self.gW_hy, self.gW_hy * 0),
               (self.gb_r, self.gb_r * 0),
               (self.gb_z, self.gb_z * 0),
               (self.gb_h, self.gb_h * 0),
               (self.gb_y, self.gb_y * 0)]
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
               (self.W_hy, self.W_hy - self.gW_hy * lr / t),
               (self.b_r, self.b_r - self.gb_r * lr / t),
               (self.b_z, self.b_z - self.gb_z * lr / t),
               (self.b_h, self.b_h - self.gb_h * lr / t),
               (self.b_y, self.b_y - self.gb_y * lr / t)]
        self.update = theano.function(
            inputs = [lr, t, tm1],
            updates = updates_w
        )
        
def rmse(py, y):
    return np.sqrt(np.mean((py - y) ** 2))

def get_pre_h(t, size, acts):
    if t == 0:
        return np.zeros((size, 1), dtype=theano.config.floatX)
    else:
        return acts["h" + str(t - 1)]

def train():
    seqs, i2w, w2i = char_sequence()

    learning_rate = 0.2;
    h_size = 100;
    
    cell = Cell((len(w2i), h_size))
    
    start = time.time()

    for i in xrange(100):
        error = 0;
        for s in xrange(len(seqs)):
            acts = {}
            e = 0;

            print i2w[np.argmax(seqs[s][0])],
            for t in xrange(len(seqs[s]) - 1):
                x = np.reshape(seqs[s][t], (len(w2i), 1))
                acts["x" + str(t)] = x
                pre_h = get_pre_h(t, h_size, acts)
                r, z, gh, h = cell.active(x, pre_h)
                acts["r" + str(t)] = r
                acts["z" + str(t)] = z
                acts["gh" + str(t)] = gh
                acts["h" + str(t)] = h

                py = cell.predict(h)
                y = np.reshape(seqs[s][t + 1], (len(w2i), 1))
                acts["py" + str(t)] = py
                acts["y" + str(t)] = y

                print i2w[np.argmax(py)],

                e = e + rmse(py, y)
            print "\nIter = " + str(i) + ", error = " + str(e / (len(seqs[s]) - 1))
            error = error + e / (len(seqs[s]) - 1)
            #print "\n"
            ##bptt
            for t in xrange(len(seqs[s]) - 2, -1, -1):
                py = acts["py" + str(t)]
                y = acts["y" + str(t)]
                r = acts["r" + str(t)]
                z = acts["z" + str(t)]
                gh = acts["gh" + str(t)]
                pre_h = get_pre_h(t, h_size, acts)
                
                if t == (len(seqs[s]) - 2):
                    post_dh = np.zeros((h_size, 1), dtype = np.float32)
                    post_dgh = np.copy(post_dh)
                    post_dr = np.copy(post_dh)
                    post_dz = np.copy(post_dh)
                    post_r = np.copy(post_dh)
                else:
                    post_dh = acts["dh" + str(t + 1)]
                    post_dgh = acts["dgh" + str(t + 1)]
                    post_dr = acts["dr" + str(t + 1)]
                    post_dz =acts["dz" + str(t + 1)]
                    post_r = acts["r" + str(t + 1)]
                
                dy, dh, dgh, dr, dz = cell.derive(y, py, r, z, gh, pre_h,
                                            post_dh, post_dgh, post_dr, post_dz, post_r)
                acts["dy" + str(t)] = dy
                acts["dh" + str(t)] = dh
                acts["dgh" + str(t)] = dgh
                acts["dr" + str(t)] = dr
                acts["dz" + str(t)] = dz
            
            ##grad
            cell.clear_grad()
            for t in xrange(len(seqs[s]) - 1):
                x = acts["x" + str(t)]
                pre_h = get_pre_h(t, h_size, acts)
                r = acts["r" + str(t)]
                h = acts["h" + str(t)]

                dy = acts["dy" + str(t)]
                dr = acts["dr" + str(t)]
                dz = acts["dz" + str(t)]
                dgh = acts["dgh" + str(t)]

                cell.grad(x, r, pre_h, h, dy, dz, dr, dgh)
           
            tm1 = t - 1
            if tm1 < 1:
                tm1 = 1
            cell.update(learning_rate, t, tm1);
        #print "\nIter = " + str(i) + ", error = " + str(error / len(seqs))
    print time.time() - start
   
if __name__ == '__main__':
    train()
