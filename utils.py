import numpy as np
import pickle as pk

# CONSTANTS
EPOCHS_DROP = 5
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.3
DROPOUT_PERCENTAGE = 0.5

MU = 0
SIGMA = 0.1
EPOCHS = 25
BATCH_SIZE = 32

PICKLES_DIR = './pickles/'


# Returns dictionary with unpickled CIFAR data
def unpickle(file):
    with open(file, 'rb') as fo:
        return pk.load(fo, encoding='bytes')


def pickle(file, data):
    with open(file, 'wb') as fo:
        return pk.dump(data, fo, pk.HIGHEST_PROTOCOL)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1. - x)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1. - x * x


def softmax(x):
    exp_max = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_max / np.sum(exp_max, axis=-1, keepdims=True)


def ReLU(x):
    return x * (x > 0)


def ReLU_prime(x):
    return 1. * (x > 0)


def cross_entropy(activation, y):
    if activation.shape == y.shape:
        one_hot_v = np.copy(y)
        return -np.mean(np.sum(np.nan_to_num(one_hot_v * np.log(activation) + (1 - one_hot_v) * np.log(1 - activation)),
                               axis=0))
    else:
        one_hot_v = np.zeros(activation.shape)
        one_hot_v[np.arange(activation.shape[0]), y] = 1
        return np.mean(np.sum(np.nan_to_num(-one_hot_v * np.log(activation) - (1 - one_hot_v) * np.log(1 - activation)),
                                         axis=1))


def cross_entropy_prime(mat, y):
    one_hot_v = np.zeros(mat.shape)
    one_hot_v[np.arange(mat.shape[0]), y] = 1
    return one_hot_v - mat

