import numpy as np
class BaseEstimator():

    fit_required = True

    def __init__(self,X,y=None,y_required = False):

        self.y_required = y_required
        X = self.transform_data(X)
        if np.size(X)==0:
            raise ValueError("Expected a matrix with rows and columns\nReceived: Empty Matrix")
        if X.ndim!=1:
            self.sample_count,self.features = X.shape[0],X.shape[1]
        else:
            self.sample_count,self.features = 1,X.shape
        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Required Argument y missing")
            y = self.transform_data(y)
            if y.size < self.sample_count:
                raise ValueError("Expected y to have length equal to X")
        self.y = y

    def transform_data(self,data):
        if type(data)!=np.array:
            data = np.array(data)
        return data            

