import numpy as np
import matplotlib.pyplot as plt
from parameters import kernel_parameters
from gauss1D import gauss1D

h, k = kernel_parameters()

def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    # samples    : DxN matrix of data points
    # k          : number of neighbors
    # Output
    # estDensity : estimated density in the range of [-5, 5]
    
    # Compute the number of the samples created
    pos = np.arange(-5, 5.0, 0.1)
    estDensity = np.zeros((pos.shape[0], 2))
    total = len(samples)
    for i, j in enumerate(pos):
        distances = np.abs(j-samples)
        distances.sort()
        radius = distances[k-1]
        volume = 2 * radius
        val = k/(total * volume)
        estDensity[i] = j, val

    return estDensity


# Generate randomised Sample data
samples = np.random.normal(0, 1, 100)

# Compute the original normal distribution
realDensity = gauss1D(0, 1, 100, 5)

estDensity = knn(samples, k)

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