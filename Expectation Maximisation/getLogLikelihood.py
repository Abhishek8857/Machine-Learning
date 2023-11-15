import numpy as np

def getLogLikelihood(means, covariances, weights, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    try:
        N, D, K = X.shape[0], X.shape[1], len(means) 
    except IndexError:
        N, D, K = 1, X.shape[0], len(means) 
           
    logLikelihood = 0

    for i in range(N):
        gauss = 0
        for j in range(K):
            if N == 1:
                mean_difference = X - means[j]
            else:
                mean_difference = X[i, :] - means[j]
            cov = covariances.T[j]
            norm_fac = 1 / (((2 * np.pi) ** (D/2)) * (np.linalg.det(cov) ** 0.5))
            val = np.exp(-0.5 * np.dot(np.dot((mean_difference).T, np.linalg.inv(cov)), (mean_difference)))
            try:
                gauss += weights[j] * norm_fac * val
            except IndexError:
                gauss += weights[0, j] * norm_fac * val
        logLikelihood += np.log(gauss)

    if type(logLikelihood) == np.ndarray:
        logLikelihood = logLikelihood[0]
        
    return logLikelihood
