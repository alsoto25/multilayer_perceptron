import numpy as np
import pickle as pk

# CONSTANTS
LEARNING_RATE = 0.1
DROPOUT_PERCENTAGE = 0.5
BATCH_SIZE = 16
EPOCHS = 100


# Returns dictionary with unpickled CIFAR data
def unpickle(file):
    with open(file, 'rb') as fo:
        return pk.load(fo, encoding='bytes')


# # ReLU function
# def relu(mat):
#     return np.maximum(mat, 0)
#
#
# def leaky_relu(mat, alpha=0.01):
#     return np.maximum(mat, -alpha * mat)
#
#
# def relu_prime(mat, y):
#     mat[mat <= 0] = 0
#     mat[mat > 0] = 1
#     return mat
#
#
# def leaky_relu_prime(mat, y, alpha=0.01):
#     dx = np.ones_like(mat)
#     dx[mat < 0] = alpha
#     return dx
#
#
# # Softmax / CrossEntropy functions
# # Snippet taken from https://deepnotes.io/softmax-crossentropy
# def softmax(x):
#     exp_max = np.exp(x - np.max(x, axis=-1, keepdims=True))
#     return exp_max / np.sum(exp_max, axis=-1, keepdims=True)
#
#
# def stable_softmax(mat):
#     return np.exp(mat - np.max(mat))/np.sum(np.exp(mat - np.max(mat)))
#
#
# def softmax_prime(mat):
#     s = mat.reshape(-1, 1)
#     return np.diagflat(s) - np.dot(s, s.T)
#
#
# def cross_entropy(p, y):
#     # return np.sum(np.nan_to_num(-y * np.log(mat) - (1 - y) * np.log(1 - mat)))
#     m = np.asarray(y).shape[0]
#     log_likelihood = -np.log(p[range(m), np.asarray(y)])
#     loss = np.sum(log_likelihood) / m
#     return loss
#
#
# def cross_entropy_prime(mat, y):
#     m = np.asarray(y).shape[0]
#     # grad = softmax(mat)
#     grad = np.copy(mat)
#     grad[range(m), y] -= 1
#     grad = grad / m
#     return grad


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1. - x)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1. - x * x


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


def ReLU(x):
    return x * (x > 0)


def ReLU_prime(x):
    return 1. * (x > 0)


def cross_entropy(activation, y):
    return -np.mean(np.sum(y * np.log(activation) + (1 - y) * np.log(1 - activation), axis=1))
