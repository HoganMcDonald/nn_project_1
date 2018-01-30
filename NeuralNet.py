import numpy as np


class NeuralNet:
    """
    a configurable plain, neural network.
    """

    def __init__(self, layer_dims):
        self.parameters = self.initialize_parameters(layer_dims)

    def initialize_parameters(self, layer_dims):
        """
        :param layer_dims: list containing the dimensions of each layer

        :return parameters: dictionary containing parameters for each layer
        """
        parameters = {}

        for layer in range(1, len(layer_dims)):
            parameters['W' + str(layer)] = parameters["W" + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
            parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1))

        return parameters

    def linear_forward(self, A, W, b):
        """
        :param A: Activations from previous layer
        :param W: weights matrix of shape (size of previous layer, size of current layer)
        :param b: bias vector (
        :return:
        """
        A = X
        layers = len(parameters)

