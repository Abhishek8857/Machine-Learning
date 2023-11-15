import numpy as np
import matplotlib.pyplot as plt
from parameters import kernel_parameters
from gauss1D import gauss1D


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]
    pos = np.arange(-5, 5.0, 0.1)
    estDensity = np.zeros((pos.shape[0], 2))
    norm_fac = 1 /(2 * np.pi * h ** 2)**(1/2)
    N = len(samples)
    
    # Code 1
    # for i in range(len(pos)):
    #     val = 0
    #     for k in samples:
    #         num = (pos[i]-k) ** 2
    #         den = 2 * h ** 2
    #         val += norm_fac * np.exp(-num/den)
    #     val /= len(samples)
    #     estDensity[i] = pos[i], val
   
    # Code 2 
    for i, j in enumerate(pos):
        val = 0
        for k in samples:
            num = (j-k) ** 2
            den = 2 * h ** 2
            val += norm_fac * np.exp(-num/den)
        val /= len(samples)
        estDensity[i] = j, val
    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created

    return estDensity


h, k = kernel_parameters()

# Generate randomised Sample data
samples = np.random.normal(0, 1, 100)

# Compute the original normal distribution
realDensity = gauss1D(0, 1, 100, 5)

estDensity = kde(samples, h)

# Plot the distribution 
fig, axes = plt.subplots(1, 1)
fig.patch.set_facecolor("black")

axes.set_facecolor("black")
axes.spines[["left", "bottom"]].set_color("white")
axes.tick_params(axis="both", colors="white")
axes.xaxis.label.set_color("white")
axes.yaxis.label.set_color("white")

plt.plot(realDensity[:, 0], realDensity[:, 1], color="red")
plt.plot(estDensity[:, 0], estDensity[:, 1], color="lightblue")

plt.show()