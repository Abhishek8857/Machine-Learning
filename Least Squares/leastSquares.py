import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    num, dim = data.shape
    ext_data = np.concatenate((np.ones((num, 1)), data), axis=1)

    result = np.dot(np.dot(np.linalg.inv(np.dot(ext_data.T, ext_data)), ext_data.T), label)
    weight, bias = result[1:], result[0]
    return weight, bias
