import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood

def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel
    skin_data = sdata
    non_skin_data = ndata
    im = img
    s_logLikelihood = np.zeros(K)
    n_logLikelihood = np.zeros(K)
    
    for i in range(K):
        # compute GMM
        s_weights, s_means, s_covariances = estGaussMixEM(skin_data, i+1, n_iter, epsilon)
        s_logLikelihood[i] = getLogLikelihood(s_means,s_covariances, s_weights, sdata)

    for i in range(K):
        # compute GMM
        n_weights, n_means, n_covariances = estGaussMixEM(non_skin_data, i+1, n_iter, epsilon)
        n_logLikelihood[i] = getLogLikelihood(n_means, n_covariances, n_weights, sdata)

    result = np.zeros_like(im)
    height, width, layers = im.shape
    
    for i  in range(height):
        for j in range(width):
            skin_likelihood = np.exp(getLogLikelihood(s_means, s_covariances, s_weights, im[i, j].T))
            non_skin_likelihood = np.exp(getLogLikelihood(n_means, n_covariances, n_weights, im[i, j].T))
            
            if skin_likelihood / non_skin_likelihood > theta:
                result[i, j] = 1

    return result

