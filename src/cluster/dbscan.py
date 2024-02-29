import numpy as np
from src.base import BaseEstimator
from .nearest_neighbors import nearest_neighbors

class DBSCAN(BaseEstimator):
    def __init__(self, eps = 0.5, min_samples = 5, metric = 'euclidean'):

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.visited = set()
        self.labels_ = None        

    def expand_cluster(self, X, idx, neighbours, label,cluster):

        self.labels_[idx] = label 

        for neighbour in neighbours:
            if neighbour not in self.visited:
                self.visited.add(neighbour)
                self.labels_[neighbour] = label
                cluster.append(neighbour)
                #find neighbours of neighbour pixel
                neighbour_neighbours = nearest_neighbors(X,X[neighbour],self.eps).tolist()
                #if neighbour has enough neighbour to become a core point, add them to the original neighbour list
                if len(neighbour_neighbours) >= self.min_samples:
                    neighbours.extend(neighbour_neighbours)

        return cluster
                    
    def fit(self,X):

        """
        This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
        long as it does this, you may change the content of this method completely and/or encapsulate the necessary
        mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """

        self.labels_ = [-1]*X.shape[0] # making an array of labels of X size for unclassified points
        clusters = []
        label =0 
        for i in range (X.shape[0]):
            if i in self.visited:
                continue # We do not check the visited nodes again
            self.visited.add(i)# appending the current index to visited
            #find neighbours of a pixel and check if it has enough neighbour pixels
            neighbours = nearest_neighbors(X,X[i],self.eps).tolist()
            if len(neighbours)<self.min_samples:
                continue
            new_cluster = []
            # consider the current pixel as core and try to expand it using its neighbours
            self.expand_cluster(X,i,neighbours,label,new_cluster)
            if len(new_cluster)>0:
                clusters.append(new_cluster)
                label = len(clusters)+1
        return self


    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_

