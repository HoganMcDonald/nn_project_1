import numpy as np


def sigmoid(Z):
    """
    :param Z: numpy array of any shape

    :return A: the sigmoid of Z
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    :param Z: numpy array of any shape
    :return A: the sigmoid of Z
    """
    A = np.maximum(0, Z)
    cache = Z

    return A, cache
