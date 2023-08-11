import pandas as pd
import numpy as np

from Network import Network


def load_data():
    """
    Loads data from csv files stored in the Data directory. The training data csv files have 60,000 rows, and the test
    data csv files have 10,000 rows.
    :return: training_data, validation_data, test_data
    """

    training_inputs = np.divide(pd.read_csv('Data/MNIST_data_train.csv').to_numpy(), 255)
    training_targets = pd.read_csv('Data/MNIST_target_train.csv').to_numpy()

    # Holds the last (zero indexed) row of a particular digit in the train csv file
    data_boundaries = []

    for i in range(1, len(training_targets)):
        if training_targets[i] != training_targets[i - 1]:
            data_boundaries.append(i - 1)
    data_boundaries.append(len(training_targets) - 1)

    training_data = []
    validation_data = []
    boundary_index = 0
    for i in range(len(training_targets)):
        if i > data_boundaries[boundary_index]:
            boundary_index += 1

        if data_boundaries[boundary_index] - 1000 < i <= data_boundaries[boundary_index]:
            validation_data.append((np.reshape(training_inputs[i], (784, 1)), training_targets[i][0]))
        else:
            training_data.append((np.reshape(training_inputs[i], (784, 1)), create_target_vector(training_targets[i])))

    test_inputs = np.divide(pd.read_csv('Data/MNIST_data_test.csv').to_numpy(), 255)
    test_targets = pd.read_csv('Data/MNIST_target_test.csv').to_numpy()

    test_data = [(np.reshape(x, (784, 1)), y[0]) for x, y in zip(test_inputs, test_targets)]

    return training_data, validation_data, test_data


def create_target_vector(target_value):
    target_vector = np.zeros((10, 1))
    target_vector[int(target_value)] = 1
    return target_vector


if __name__ == '__main__':
    training_data, validation_data, test_data = load_data()

    net = Network.Network([784, 100, 10])

    net.sgd(training_data, 60, 10, 0.05, 5, test_data=test_data)
