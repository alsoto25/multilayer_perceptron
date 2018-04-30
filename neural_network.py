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


# Layer containing all w, with their respective weights and activation functions, along with their derivative.
class Layer:
    """
    Constructor for a Layer object.
    <params> shape -> Contains the shape of the Layer (input size, number of layers)
             activation_function -> Activation function to use for the layer
             prime_activation_function -> Derivative of the activation function
    """
    def __init__(self, shape, activation_function=ut.relu, prime_activation_function=ut.relu_prime):
        self.z = np.ones(shape[1])
        self.output = np.ones(shape[1])
        self.prime_output = np.ones(shape[0])
        self.w = np.random.standard_normal(shape)

        self.activation_function = activation_function
        self.prime_activation_function = prime_activation_function

    def set_w(self, w):
        self.w = w

    def feed_forward_layer(self, X):
        self.z = np.dot(X, self.w)
        self.output = self.activation_function(self.z / np.max(self.z))

    def back_prop_layer(self, prev_layer, y):
        self.prime_output = np.dot(last_activation.T, delta)

    def update_layer(self, input_matrix):
        self.w -= ut.LEARNING_RATE * np.dot(input_matrix.T, self.prime_output)


class Network:

    def __init__(self, hidden_layer_size_list, activation_functions_list, prime_functions_list):
        self.class_amount = np.unique(test_lbls).size
        self.cost = 0

        self.layers = np.asarray([Layer((test_imgs.shape[1], hidden_layer_size_list[0]),
                                        activation_function=activation_functions_list[0],
                                        prime_activation_function=prime_functions_list[0])])

        for i in range(1, len(hidden_layer_size_list)):
            self.layers = np.append(self.layers,
                                    [Layer((hidden_layer_size_list[i-1], hidden_layer_size_list[i]),
                                     activation_function=activation_functions_list[i],
                                     prime_activation_function=prime_functions_list[i])], axis=0)

        self.layers = \
            np.append(self.layers,
                      [Layer((hidden_layer_size_list[len(hidden_layer_size_list)-1], self.class_amount),
                             activation_function=activation_functions_list[len(activation_functions_list) - 1],
                             prime_activation_function=prime_functions_list[len(activation_functions_list) - 1])])

    def feed_forward_batch(self, x_batch):
        for i in range(self.layers.size):
            if i == 0:
                self.layers[i].feed_forward_layer(x_batch)
            else:
                self.layers[i].feed_forward_layer(self.layers[i-1].output)

    def cost(self, mat, y_batch):
        return ut.cross_entropy(mat, y_batch)

    def back_prop_batch(self, y_batch):
        self.layers[-1].back_prop_layer(1, y_batch)

        for i in range(2, self.layers.size + 1):
            error = np.dot(self.layers[-i+1].prime_output, self.layers[-i+1].w.T)
            self.layers[-i].back_prop_layer(error, y_batch)

    def update_batch(self, x_batch):
        self.layers[0].update_layer(x_batch)

        for i in range(1, self.layers.size):
            self.layers[i].update_layer(self.layers[i-1].output)

    def eval(self, x_batch, y_batch):
        self.feed_forward_batch(x_batch)

        test_results = [(x, y) for x, y in zip(np.argmax(self.layers[-1].output, axis=1), y_batch)]
        return sum(int(x == y) for (x, y) in test_results)
