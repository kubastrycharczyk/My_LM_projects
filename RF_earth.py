import pandas as pd
import numpy as np

dataframe=pd.read_csv("dataset.csv")
target_name = "Magnitude"


from sklearn.model_selection import train_test_split
from sklearn import preprocessing


y = dataframe[target_name]
x = dataframe.drop(target_name, axis=1)
print(x)

class Forest_Predictor:
    def standarizer(z):
        minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        z=minmax_scaler.fit_transform(z)
  
    def __init__(self, dataframe, target_name):
        y = dataframe[target_name].to_numpy()
        X = dataframe.drop(target_name, axis=1).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, train_size = 0.8)
        self.standarizer(self.X_test)
        self.standarizer(self.X_train)
        self.standarizer(self.y_test)
        self.standarizer(self.y_train)

    def setting_parameters():
        pass

    def train_forest():
        pass
    
    def predict_forest():
        pass

