import numpy as np
from getLogLikelihood import getLogLikelihood
from EStep import EStep

def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).
    
    N = X.shape[0] # 275
    D = X.shape[1] # 2
    K = gamma.shape[1] # 3
    
    logLikelihood = np.zeros((1, K))
    means = np.zeros((K, D))
    weights = np.zeros((1, K))
    covariances = np.zeros((D, D, K))

    for j in range(K):
        weight_sum = 0
        mean_sum = 0 
        cov_sum = 0
        
        for i in range(N):
            weight_sum += gamma[i, j]
        weights[0, j] = weight_sum / N
        
        for i in range(N):
            mean_sum += gamma[i, j] * X[i]
        means[j, :] = mean_sum / weight_sum

        for i in range(N):
            cov_sum += gamma[i, j] * np.outer(X[i] - means[j], (X[i] - means[j]).T)
        covariances.T[j] = cov_sum / weight_sum
    
    logLikelihood = getLogLikelihood(means, covariances, weights, X)

    return weights, means, covariances, logLikelihood
