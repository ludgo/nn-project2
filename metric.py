import numpy as np
from scipy.spatial.distance import pdist


def L_1(u, v):
    # minkowski p=1
    return pdist(np.asarray([u, v]), 'cityblock')[0] # =manhattan

def L_2(u, v):
    # minkowski p=2
    return pdist(np.asarray([u, v]), 'euclidean')[0]

def L_max(u, v):
    # minkowski p=inf
    return pdist(np.asarray([u, v]), 'chebyshev')[0]
