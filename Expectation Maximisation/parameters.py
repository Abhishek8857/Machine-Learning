def EM_parameters():
    epsilon = 0.0001 # regularization
    K = 3 # number of desired clusters
    n_iter = 5  # number of iterations
    skin_n_iter = 10
    skin_epsilon = 0.0008
    skin_K = 5
    theta = 5.0  # threshold for skin detection

    return epsilon, K, n_iter, skin_n_iter, skin_epsilon, skin_K, theta
