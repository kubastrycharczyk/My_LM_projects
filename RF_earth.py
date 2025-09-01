import pandas as pd
import numpy as np

dataframe=pd.read_csv("dataset.csv")
target_name = "Magnitude"


from sklearn.model_selection import train_test_split


y = dataframe[target_name]
x = dataframe.drop(target_name, axis=1)
print(x)

class Predictor:
  
    def __init__(self, dataframe, target_name):
        y = dataframe[target_name].to_numpy()
        x = dataframe.drop(target_name, axis=1).to_numpy()
          self.X_test = None
    self.X_train = None
    self.
    def predict():
        pass