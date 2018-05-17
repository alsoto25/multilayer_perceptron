import numpy as np
import pickle as pk

# CONSTANTS
EPOCHS_DROP = 10
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.5
DROPOUT_PERCENTAGE = 0.5

MU = 0
SIGMA = 0.001
EPOCHS = 100
BATCH_SIZE = 50

PICKLES_DIR = './pickles/'


# Turn array greyscale
def greyscale_image(arr):
    vectorized_greyscale_img = np.vectorize(greyscale_pixel)

    res = vectorized_greyscale_img(arr[:, :, 0], arr[:, :, 1], arr[:, :, 2])

    return res


def greyscale_pixel(r, g, b):
    return int(round(0.2989 * r + 0.5870 * g + 0.1140 * b))


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

        fixed_section = np.nan_to_num((1 - one_hot_v) * np.log(1 - activation))

        return -np.mean(np.sum(one_hot_v * np.log(activation) + fixed_section, axis=1))


def cross_entropy_prime(mat, y):
    one_hot_v = np.zeros(mat.shape)
    one_hot_v[np.arange(mat.shape[0]), y] = 1
    return one_hot_v - mat

