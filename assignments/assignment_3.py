import math
import os
from math import tanh

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import mean_squared_error


def plot_training_data(X, Y):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Y, cmap='bwr')
    plt.axhline(y=0.5, color='grey', linestyle='--')
    plt.axvline(x=0.5, color='grey', linestyle='--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    plt.close()


def generate_data(n, label):
    X = []
    Y = []

    for x1, x2 in zip(np.random.randint(low=0, high=100, size=n), np.random.randint(low=0, high=100, size=n)):
        if label == 'AND':
            if x1 >= 50 and x2 >= 50:
                Y.append(1)
            else:
                Y.append(0)
        if label == 'OR':
            if x1 >= 50 or x2 >= 50:
                Y.append(1)
            else:
                Y.append(0)
        if label == 'XOR':
            if (x1 >= 50 > x2) or (x1 < 50 <= x2):
                Y.append(1)
            else:
                Y.append(0)

        X.append([x1/100, x2/100])

    return np.asarray(X), np.asarray(Y)


def train(label):
    np.random.seed(1)
    training_inputs, labels = generate_data(100, label)

    plot_training_data(training_inputs, labels)

    # TODO: implement


train(label='AND')
