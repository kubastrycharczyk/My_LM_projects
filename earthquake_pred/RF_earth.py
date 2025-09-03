import pandas as pd
import numpy as np

dataframe=pd.read_csv("dataset.csv")
target_name = "Magnitude"


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier


y = dataframe[target_name]
x = dataframe.drop(target_name, axis=1)
print(x)

class Forest_Predictor:
    def standarizer(z):
        minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        return minmax_scaler.fit_transform(z)
  
    def __init__(self, dataframe, target_name):
        y = dataframe[target_name].to_numpy()
        X = dataframe.drop(target_name, axis=1).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, train_size = 0.8)
        self.X_test = self.standarizer(self.X_test)
        self.X_train = self.standarizer(self.X_train)

        self.n_jobs=-1 #how many processors in usage
        self.random_state = 0 
        self.n_esimators = 100


    def setting_parameters(self,n_jobs, random_state, n_estimators):
        self.n_jobs=n_jobs
        self.random_state = random_state
        self.n_esimators = n_estimators

    def train_forest(self):
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs, n_estimators=self.n_esimators).fit(self.X_train, self.y_train)
        score = cross_val_score(self.model, self.X, self.y, cv=5, scoring="accuracy")

    def test_forest(self):
        prediction=self.model.predict(self.X_test)
        predictio=prediction-self.y_test
        return prediction
    
    def predict_forest(self, exa):
        self.model.predict(exa) 

