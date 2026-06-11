import numpy as np
import pandas as pd

class LinearRegression:
    
    def __init__(self):
        self.p = None
        self.m = {}

    def encode(self, arr, column):
        arr = arr.copy()
        if column not in self.m:
            self.m[column] = {v: i for i, v in enumerate(arr[column].unique())}
        arr[column] = arr[column].map(self.m[column])          
        return arr         
    def test_train(self, d):
        total = len(d)
        tr = int(total * 70 / 100)
        td = d[:tr]
        ts = d[tr:]   
        return td, ts                
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        self.p = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.array(X) 
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        return X @ self.p