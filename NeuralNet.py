import numpy as np
from nn_helpers import sigmoid, relu


class NeuralNet:
    """
    a configurable plain, neural network.
    """

    def __init__(self, layer_dims):
        self.parameters = self.initialize_parameters(layer_dims)
        self.cache = ()


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
        :param b: bias vector of shape (size of the current layer, 1)

        :return Z: pre-activation parameter
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        :param A_prev: activations from previous layer
        :param W: weights matrix
        :param b: bias vector
        :param activation: the method of activation | relu or sigmoid

        :return A: output matrix of the activation function
        """
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        else:
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        return A

    def forward_propagation(self, X, parameters):
        """
        performs all steps in the forward propagation of the network

        :param X: data, numpy array of shape (input size, number of examples)
        :param parameters: initialized parameters

        :return AL: last post-activation value
        :return caches: list of caches for back propagation
        """
        caches = []
        A = X
        layers = len(parameters)

        for layer in range(1, layers):
            A_prev = A

            A, cache = self.linear_activation_forward(A_prev,
                                                      parameters['W' + str(layer)],
                                                      parameters["b" + str(layer)],
                                                      activation='relu')
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A,
                                              parameters['W' + str(layers)],
                                              parameters['b' + str(layers)],
                                              activation='sigmoid')
        caches.append(cache)