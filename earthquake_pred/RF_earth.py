import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


class Forest_Predictor:
    def create_standarizer(self, z):
        self.minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        return self.minmax_scaler.fit_transform(z)
    
    def standarize(self,z):
            return self.minmax_scaler.transform(z)

  
    def __init__(self, dataframe, target_name):
        y = dataframe[target_name].to_numpy()
        X = dataframe.drop(target_name, axis=1).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, train_size = 0.8)
        self.X_train = self.create_standarizer(self.X_train)
        self.X_test = self.standarize(self.X_test)

        self.n_jobs=-1 #how many processors in usage
        self.random_state = 0 
        self.n_estimators = 100


    def setting_parameters(self,n_jobs, random_state, n_estimators):
        self.n_jobs=n_jobs
        self.random_state = random_state
        self.n_estimators = n_estimators

    def train_forest(self):
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs, n_estimators=self.n_estimators).fit(self.X_train, self.y_train)
        score = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring="accuracy")
        return score
    
    def test_forest(self):
        prediction=self.model.predict(self.X_test)
        return prediction
    
    def predict_forest(self, exa):
        exa = self.standarize(exa)
        return self.model.predict(exa) 

