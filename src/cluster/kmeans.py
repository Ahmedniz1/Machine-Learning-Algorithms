import numpy as np
import os

from src.base import BaseEstimator 
from .nearest_neighbors import cluster_neighbors
from src.metrics import euclidean_distance
class KMeans(BaseEstimator):
    def __init__(self, n_clusters=8,n_init='auto', max_iter=100):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
    
    def _init_centroids(self,X):
        self.centroids = X[np.random.choice(len(X),self.n_clusters, replace = False)]

    def fit(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Computes the K-Mean clustering.
        """
        self.predict(X)
        return self.labels_
    
    def predict(self,X):
        """
        Computes cluster centers and predicts cluster label for each sample
        """
        self._init_centroids(X)
        for _ in range(self.max_iter):
            # compute distances of each pixel against each center point -> 2d list 

            self.labels_ = cluster_neighbors(X,self.centroids)
            new_centroids = np.array([np.mean(X[np.where(self.labels_ == i)],axis = 0) for i in range(self.n_clusters)])
            if np.array_equal(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        return self
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
