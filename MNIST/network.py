# Libraries
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):

        # sizes is list with number of neurons in each layer
        # e.g. sizes = [2,3,1] is network with three layers, 1st with 2 neurons etc.
        self.num_layers = len(sizes)
        self.sizes = sizes

        # input layer has no bias
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # weight connecting 1st to 2nd, 2nd to 3rd, ... last second to last
        self.weights = [np.random.randn(y, x)for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):

        # return  output of network for input a
        # apply sigmoid function for each layer
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        # train NN using mini-batch SGD
        # training_data = list of tuples 9x,y) representing training inputs and desired outputs
        # test_data = if provided, network will be evaluated against test data after each epoch

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):

        # update network's weights and biases
        # apply gradient descent using backpropagation to a single mini batch
        # mini_batch = list of tuples (x,y)
        # eta = learning rate

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        # return tuple representing gradient for cost function
        # first element in tuple are layer-by-layer lists of numpy arrays

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        # return number of test inputs for which outputs are correct
        # output is index of neuron in final layer with highest activation
        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):

        # return vector of partiaul derivatives
        # partial of cost function / partial for output activations
        
        return (output_activations-y)

# Miscellaneous functions
def sigmoid(z):
    
    # sigmoid function
    # z is a vector or numpy array
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    # derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))