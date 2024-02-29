import numpy as np
from scipy.spatial.distance import cdist
def euclidean_distance(X, y):
    if y.ndim!=2:
        y = [y]
    return cdist (X,y)
    # elif isinstance(y, np.ndarray):
    #     # Case 3: x is a point and y is an array
    #     return np.linalg.norm(y - X, ord = 2, axis=1)
    # else:
    #     # Case 4: Both x and y are points
    #     return np.linalg.norm(y - X, ord = 2)
    