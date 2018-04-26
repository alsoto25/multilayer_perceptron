import numpy as np
import pickle as pk


# Returns dictionary with unpickled CIFAR data
def unpickle(file):
    with open(file, 'rb') as fo:
        return pk.load(fo, encoding='bytes')

# ReLU function
def relu(mat):
    return np.maximum(mat, 0)


def relu_prime(mat):
    mat[mat <= 0] = 0
    mat[mat > 0] = 1
    return mat


# Softmax / CrossEntropy functions
# Snippet taken from https://deepnotes.io/softmax-crossentropy
def softmax(mat):
    exps = np.exp(mat)
    return exps / np.sum(exps)


def stable_softmax(mat):
    exps = np.exp(mat - np.max(mat))
    return exps / np.sum(exps)


def cross_entropy(mat):
    p = softmax(mat)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def cross_entropy_prime(mat, y):
    """
    mat is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(mat)
    grad[range(m), y] -= 1
    grad = grad/m
    return grad
