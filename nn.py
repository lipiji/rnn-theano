#pylint: skip-file

class Layer(object):
    def __init__(self, shape):
        pass

    def active(self, X):
        pass

    def calculate_delta(self, propagation = None, Y = None):
        pass

    def update(self, X, lr):
        pass
 
class NN(object):
    def __init__(self, layers):
        self.layers = layers

    def batch_train(self, X, Y, lr):
        self.feed_forward(X)
        self.back_propagarion(Y)
        self.update_parameters(X, lr)

    def feed_forward(self, X):
        for i in xrange(len(self.layers)):
            if i == 0:
                self.layers[i].active(X)
            else:
                self.layers[i].active(self.layers[i - 1].activation)
    
    def back_propagarion(self, Y):
        for i in xrange(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].calculate_delta(None, Y)
            else:
                self.layers[i].calculate_delta(self.layers[i + 1].propagation, None)
    
    def update_parameters(self, X, lr):
        for i in xrange(len(self.layers)):
            if i == 0:
                self.layers[i].update(X, lr)
            else:
                self.layers[i].update(self.layers[i - 1].activation, lr)
    
    def output(self):
        return self.layers[len(self.layers) - 1].activation

    def predict(self, x):
        self.feed_forward(x)
        return self.output()

