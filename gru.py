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
       
        ###

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

           
    def active(self, x, pre_h):
        r = T.nnet.sigmoid(T.dot(self.W_xr, x) + T.dot(self.W_hr, pre_h) + self.b_r)
        z = T.nnet.sigmoid(T.dot(self.W_xz, x) + T.dot(self.W_hz, pre_h) + self.b_z)
        gh = T.tanh(T.dot(self.W_xh, x) + T.dot(self.W_hh, r * pre_h) + self.b_h)
        h = z * pre_h + (1 - z) * gh
        return r, z, gh, h 

    def predict(self, h):
        y = T.nnet.softmax((T.dot(self.W_hy, h) + self.b_y).T).T
        return y

    def derive(self, y, py, r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz, post_r):
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
    
    x = T.matrix("x")
    pre_h = T.matrix("pre_h")
    active = theano.function(
        inputs = [x, pre_h],
        outputs = cell.active(x, pre_h)
    )

    h = T.matrix("h")
    predict = theano.function(
        inputs = [h],
        outputs = cell.predict(h)
    )

    y, py, r, z, gh, post_dh, post_dgh, post_dr, post_dz, post_r = \
        T.matrices("y", "py", "r", "z", "gh", "post_dh", "post_dgh", "post_dr", "post_dz", "post_r")
    derive = theano.function(
        inputs = [y, py, r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz, post_r],
        outputs = cell.derive(y, py, r, z, gh, pre_h, post_dh, post_dgh, post_dr, post_dz, post_r)
    )

    dy, dz, dr, dgh = T.matrices("dy", "dz", "dr", "dgh")
    updates_grad = [(cell.gW_xr, cell.gW_xr + T.dot(dr, x.T)),
               (cell.gW_xz, cell.gW_xz + T.dot(dz, x.T)),
               (cell.gW_xh, cell.gW_xh + T.dot(dgh, x.T)),
               (cell.gW_hr, cell.gW_hr + T.dot(dr, pre_h.T)),
               (cell.gW_hz, cell.gW_hz + T.dot(dz, pre_h.T)),
               (cell.gW_hh, cell.gW_hh + T.dot(dgh, (r * pre_h).T)),
               (cell.gW_hy, cell.gW_hy + T.dot(dy, h.T)),
               (cell.gb_r, cell.gb_r + dr),
               (cell.gb_z, cell.gb_z + dz),
               (cell.gb_h, cell.gb_h + dgh),
               (cell.gb_y, cell.gb_y + dy)]
    grad = theano.function(
        inputs = [x, r, pre_h, h, dy, dz, dr, dgh],
        updates = updates_grad
    )

    ## how to clear the content?
    updates_clear = [(cell.gW_xr, cell.gW_xr * 0),
               (cell.gW_xz, cell.gW_xz * 0),
               (cell.gW_xh, cell.gW_xh * 0),
               (cell.gW_hr, cell.gW_hr * 0),
               (cell.gW_hz, cell.gW_hz * 0),
               (cell.gW_hh, cell.gW_hh * 0),
               (cell.gW_hy, cell.gW_hy * 0),
               (cell.gb_r, cell.gb_r * 0),
               (cell.gb_z, cell.gb_z * 0),
               (cell.gb_h, cell.gb_h * 0),
               (cell.gb_y, cell.gb_y * 0)]
    clear_grad = theano.function(
        inputs = [],
        updates = updates_clear
    )

    lr = T.scalar()
    t = T.scalar()
    tm1 = T.scalar()
    updates_w = [(cell.W_xr, cell.W_xr - cell.gW_xr * lr / t),
               (cell.W_xz, cell.W_xz - cell.gW_xz * lr / t),
               (cell.W_xh, cell.W_xh - cell.gW_xh * lr / t),
               (cell.W_hr, cell.W_hr - cell.gW_hr * lr / tm1),
               (cell.W_hz, cell.W_hz - cell.gW_hz * lr / tm1),
               (cell.W_hh, cell.W_hh - cell.gW_hh * lr / tm1),
               (cell.W_hy, cell.W_hy - cell.gW_hy * lr / t),
               (cell.b_r, cell.b_r - cell.gb_r * lr / t),
               (cell.b_z, cell.b_z - cell.gb_z * lr / t),
               (cell.b_h, cell.b_h - cell.gb_h * lr / t),
               (cell.b_y, cell.b_y - cell.gb_y * lr / t)]
    update = theano.function(
        inputs = [lr, t, tm1],
        updates = updates_w
    )

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
                r, z, gh, h = active(x, pre_h)
                acts["r" + str(t)] = r
                acts["z" + str(t)] = z
                acts["gh" + str(t)] = gh
                acts["h" + str(t)] = h

                py = predict(h)
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
                
                dy, dh, dgh, dr, dz = derive(y, py, r, z, gh, pre_h,
                                            post_dh, post_dgh, post_dr, post_dz, post_r)
                acts["dy" + str(t)] = dy
                acts["dh" + str(t)] = dh
                acts["dgh" + str(t)] = dgh
                acts["dr" + str(t)] = dr
                acts["dz" + str(t)] = dz
            
            ##grad
            clear_grad()
            for t in xrange(len(seqs[s]) - 1):
                x = acts["x" + str(t)]
                pre_h = get_pre_h(t, h_size, acts)
                r = acts["r" + str(t)]
                h = acts["h" + str(t)]

                dy = acts["dy" + str(t)]
                dr = acts["dr" + str(t)]
                dz = acts["dz" + str(t)]
                dgh = acts["dgh" + str(t)]

                grad(x, r, pre_h, h, dy, dz, dr, dgh)
           
            tm1 = t - 1
            if tm1 < 1:
                tm1 = 1
            update(learning_rate, t, tm1);
        #print "\nIter = " + str(i) + ", error = " + str(error / len(seqs))
    print time.time() - start
   
if __name__ == '__main__':
    train()
