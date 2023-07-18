import pandas as pd
import numpy as np

from Network import Network


def create_target_vector(target_value):
    target_vector = np.zeros((10, 1))
    target_vector[int(target_value)] = 1
    return target_vector


if __name__ == '__main__':
    training_inputs = np.divide(pd.read_csv('Data/MNIST_data_train.csv').to_numpy(), 255)
    training_targets = pd.read_csv('Data/MNIST_target_train.csv').to_numpy()

    training_data = [(np.reshape(x, (784, 1)), create_target_vector(y))
                     for x, y in zip(training_inputs, training_targets)]

    # print(training_data[0])

    test_inputs = np.divide(pd.read_csv('Data/MNIST_data_test.csv').to_numpy(), 255)
    test_targets = pd.read_csv('Data/MNIST_target_test.csv').to_numpy()

    test_data = [(np.reshape(x, (784, 1)), y[0]) for x, y in zip(test_inputs, test_targets)]

    # print(test_data[0])

    net = Network.Network([784, 100, 10])

    net.sgd(training_data, 50, 50, 0.025, test_data=test_data)
