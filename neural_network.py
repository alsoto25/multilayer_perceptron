import utils as ut
import numpy as np

from mnist import MNIST


# MNIST Constants
mndata = MNIST('./MNIST')
test_imgs, test_lbls = mndata.load_testing()
train_imgs, train_lbls = mndata.load_training()

train_imgs = np.asarray(train_imgs)
train_lbls = np.asarray(train_lbls)

test_imgs = np.asarray(test_imgs)
test_lbls = np.asarray(test_lbls)


# Layer containing all neurons, with their respective weights and activation functions, along with their derivative.
class Layer:
    """
    Constructor for a Layer object.
    <params> shape -> Contains the shape of the Layer (input size, number of layers)
             activation_function -> Activation function to use for the layer
             prime_activation_function -> Derivative of the activation function
    """
    def __init__(self, shape, activation_function=ut.cross_entropy, prime_activation_function=ut.cross_entropy_prime):
        self.output = np.ones(shape[1])
        self.prime_output = np.ones(shape[0])
        self.neurons = np.random.standard_normal(shape)

        self.activation_function = activation_function
        self.prime_activation_function = prime_activation_function

    def set_neurons(self, neurons):
        self.neurons = neurons

    def feed_forward_layer(self, X):
        s = np.dot(self.neurons, X)
        self.output = self.activation_function(s)

    def back_prop_layer(self, error, y):
        self.prime_output = error * self.prime_activation_function(y)

    def update_layer(self, ):