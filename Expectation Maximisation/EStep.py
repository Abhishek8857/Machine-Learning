import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    logLikelihood = getLogLikelihood(means, covariances, weights, X)
    
    N = X.shape[0]     
    D = X.shape[1]  
    K = len(means) 
    gamma = np.zeros((N, K))
    # Denominator
    for i in range(N):
        den = 0
        for j in range(K):
            mean_difference = X[i] - means[j]
            norm_fac = 1 / (((2 * np.pi) ** (D/2)) * (np.linalg.det(covariances.T[j]) ** 0.5))
            val = np.exp(-0.5 * np.dot(np.dot((mean_difference).T, np.linalg.inv(covariances.T[j])), (mean_difference)))            
            try:
                if len(weights) > 1:
                    den += norm_fac * val * weights[j]
                else:
                    den += norm_fac * val * weights[0][j]
            except IndexError:
                    den += norm_fac * val * weights[0]
                
                    
    # Numerator
        for j in range(K):
            norm_fac = 1 / (((2 * np.pi) ** (D/2)) * (np.linalg.det(covariances.T[j]) ** 0.5))
            val = np.exp(-0.5 * np.dot(np.dot((X[i] - means[j]).T, np.linalg.inv(covariances.T[j])), (X[i] - means[j])))
            try:
                if len(weights) > 1:
                    num = norm_fac * val * weights[j]
                else:
                    num = norm_fac * val * weights[0][j]
            except IndexError:
                    num = norm_fac * val * weights[0]
                
            gamma[i, j] = num/den
    return [logLikelihood, gamma]

