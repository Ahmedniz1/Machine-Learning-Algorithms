import numpy as np

from src.metrics import euclidean_distance 

def nearest_neighbors( data, datapoint, dist, return_index = True):
    distance = euclidean_distance(data , datapoint)
    return np.where(distance<= dist)[0]


def cluster_neighbors(X,centers):

    distances = euclidean_distance(X, centers)  
    cluster_id = [np.argmin(i) for i in distances]
    return np.array(cluster_id)