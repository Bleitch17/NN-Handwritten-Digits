import numpy as np
import random


class Network:
    def __init__(self, layer_sizes):
        """
        Constructor for the neural network.
        :param layer_sizes: a list of numbers, where layer_sizes[i] represents the number of neurons in the network
        in layer i. For example, [2, 3, 2] will create a network with 2 input neurons, 3 neurons in the hidden layer,
        and 2 output neurons
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        self.biases = [np.random.randn(layer_size, 1) for layer_size in layer_sizes[1:]]
        self.weights = [np.random.randn(layer_size, prev_layer_size)
                        for layer_size, prev_layer_size in zip(layer_sizes[1:], layer_sizes[:-1])]

    def feedforward(self, activations):
        """
        Calculates the output of the network for a given input
        :param activations: the input to the network, i.e.: the output of the input layer. Must have the same number of
        elements as the number of neurons in the input layer.
        :return: the activations of the output layer.
        """
        for b, w in zip(self.biases, self.weights):
            activations = sigmoid(np.dot(w, activations) + b)
        return activations

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Trains the network using stochastic gradient descent.
        :param training_data: a list of (x, y) tuples, where x is the input (activations of the input layer) and y is
        the correct output (activations of the output layer).
        :param epochs: the number of training epochs.
        :param mini_batch_size: the size of each mini batch.
        :param eta: the learning rate.
        :param test_data: optional - a list of (x, y) tuples, where x is the input (activations of the input layer) and
        y is the correct output digit (NOT the activations of the output layer).
        :return: nothing - mutates the biases and weights in the network.
        """
        n = len(training_data)
        n_test = len(test_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch_to_matrix_tuple(mini_batch), eta)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)}/{n_test}")
            else:
                print(f"Epoch {epoch} complete")

    def evaluate(self, test_data):
        """
        Feeds the test data into the network.
        :param test_data: a list of (x, y) tuples, where x is the input (activations of the input layer) and
        y is the correct output digit (NOT the activations of the output layer).
        :return: The total number of correct outputs from the test_data.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the biases and weights in the network for each mini batch, based on the results from the backpropagation
        algorithm.
        :param mini_batch: a list of (x, y) tuples, where x is the input (activations of the input layer) and y is
        the correct output (activations of the output layer).
        :param eta: the learning rate.
        :return: nothing - mutates the biases and weights in the network.
        """
        nabla_b, nabla_w = self.backprop(mini_batch[0], mini_batch[1])

        # Note - the loop above sums the deltas for each (x, y) in the mini batch, so need to take the average
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Executes the backpropagation algorithm.
        :param x: the input (activations of the input layer)
        :param y: the correct output (activations of the output layer)
        :return: a tuple (delta_nabla_b, delta_nabla_w), where delta_nabla_b is the list of vectors containing the
        amounts by which the biases in each layer should change, and delta_nabla_w is the list of
        matrices containing the amounts by which the weights in each later should change.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Note - no need for a z vector for the input layer
        activation = x
        activations = [x]
        z_matrices = []

        # Feedforward + tracking intermediate values for use in later calculations
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_matrices.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Error from the output layer
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(z_matrices[-1])
        nabla_b[-1] = np.sum(delta, axis=1).reshape(delta.shape[0], 1)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # For a given layer, find its error by using the error from the layer after it - since the output layer error is
        #  known, work backwards
        for layer in range(2, self.num_layers):
            z = z_matrices[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = np.sum(delta, axis=1).reshape(delta.shape[0], 1)
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w


def cost_derivative(output_activations, y):
    return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1 * z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def mini_batch_to_matrix_tuple(mini_batch):
    return np.concatenate([x[0] for x in mini_batch], axis=1), np.concatenate([x[1] for x in mini_batch], axis=1)
