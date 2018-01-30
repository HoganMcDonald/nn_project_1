# Tests for neural net methods
import time
import NeuralNet as nn

test_layer_dims = [5, 4, 3]


# instantiate class and initialize parameters

start = time.clock()
test_net = nn.NeuralNet(test_layer_dims)
end = time.clock()

for layer in range(1, len(test_layer_dims)):
    assert (test_net.parameters['W' + str(layer)].shape == (test_layer_dims[layer], test_layer_dims[layer - 1]))
    assert (test_net.parameters['b' + str(layer)].shape == (test_layer_dims[layer], 1))

print("\nNeuralNet.initialize_parameters passed | {:.2f}ms".format((end - start) * 1000))


# linear forward on instantiated network
