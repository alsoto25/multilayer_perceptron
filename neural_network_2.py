#### Libraries
# Standard library
import random
import utils as ut
import numpy as np

from mnist import MNIST

# MNIST Constants
mndata = MNIST('./MNIST')
test_imgs, test_lbls = mndata.load_testing()
train_imgs, train_lbls = mndata.load_training()

test_data = [(x, y) for x, y in zip(test_imgs, test_lbls)]
train_data = [(x, y) for x, y in zip(train_imgs, train_lbls)]


# Code based on Michael Nielsen's Github
# https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
class Network(object):

    def __init__(self, layer_size_list, activation_functions_list, prime_functions_list):
        self.num_layers = len(layer_size_list)
        self.activation_functions_list = activation_functions_list
        self.prime_functions_list = prime_functions_list
        self.sizes = layer_size_list
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def test_feed_forward(self, mat):
        for bias, weight, activation in zip(self.biases, self.weights, self.activation_functions_list):
            mat = activation(np.dot(weight, mat) + bias)
        return mat

    def feed_forward_batch(self, batch):
        x = np.asarray([e[0] for e in batch])
        x = x / np.max(x)

        activation = x
        activations = [x]
        zs = []
        for bias, weight, activation_func in zip(self.biases, self.weights, self.activation_functions_list):
            z = np.dot(activation, weight) + bias
            z = z
            zs.append(z)

            activation = activation_func(z)
            activations.append(activation)
        return zs, activations

    def back_prop_batch(self, batch, zs, activations):
        y = [e[1] for e in batch]

        bias_prime_output = [np.zeros(b.shape) for b in self.biases]
        weight_prime_output = [np.zeros(w.shape) for w in self.weights]

        delta = self.prime_functions_list[-1](activations[-1], y)
        bias_prime_output[-1] = delta
        weight_prime_output[-1] = np.dot(activations[-2].transpose(), delta)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.prime_functions_list[-l](z, y)
            delta = np.dot(delta, self.weights[-l+1].transpose()) * sp

            bias_prime_output[-l] = delta
            weight_prime_output[-l] = np.dot(activations[-l-1].transpose(), delta)

        return bias_prime_output, weight_prime_output

    def update_batch(self, batch, bias_prime_output, weight_prime_output):
        average_bias_prime = [np.zeros(b.shape) for b in self.biases]
        average_weight_prime = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            average_bias_prime = [nb + dnb for nb, dnb in zip(average_bias_prime, bias_prime_output)]
            average_weight_prime = [nw + dnw for nw, dnw in zip(average_weight_prime, weight_prime_output)]

        self.weights = [w - (ut.LEARNING_RATE / len(batch)) * nw
                        for w, nw in zip(self.weights, average_weight_prime)]
        self.biases = [b - (ut.LEARNING_RATE / len(batch)) * nb
                       for b, nb in zip(self.biases, average_bias_prime)]

    def cost(self, mat, batch):
        y = [e[1] for e in batch]
        return ut.cross_entropy(mat, y)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.test_feed_forward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.test_feed_forward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
