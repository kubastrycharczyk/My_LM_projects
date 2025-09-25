import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer,mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

class Forest_Predictor:
    def __init__(self, dataframe, target_name, model_name="RandomForest"):
        y = dataframe[target_name].to_numpy()
        X = dataframe.drop(target_name, axis=1).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, train_size = 0.8, random_state=42)
        self.trans=None
        self.random_state = 0 

        
        self.model_name = model_name
        self.model_params = {}
        self.model = None
        self.models = {
            "RandomForest: ": RandomForestRegressor,
            "LightGBM: ": LGBMRegressor
        }
        self.setters = {
            "RandomForest: ": self._set_params_forest,
            "LightGBM: ": self._set_params_lightgbm
        }
    
    def _set_model(self):
        model_class = self.models.get(self.model_name)
        if model_class is None:
            raise ValueError(f"Nieznany model: {self.model_name}")
        self.model = model_class(**self.model_params)
    
    def create_standardizer(self, z):
        self.minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        return self.minmax_scaler.fit_transform(z)
    
    def standardizer(self,z):
            return self.minmax_scaler.transform(z)
    
    def standardize(self):
        self.X_train = self.create_standardizer(self.X_train)
        self.X_test = self.standardizer(self.X_test)

    def make_polynomial(self, degree=2):
        trans = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)    
        self.X_train=trans.fit_transform(self.X_train)
        self.X_test=trans.transform(self.X_test)
        self.trans =trans
    
    def _set_params_forest(self, **kwargs):
        defaults = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
        }
        self.model_params.update(defaults)
        self.model_params.update(kwargs)

    def _set_params_lightgbm(self, **kwargs):
        defaults = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        self.model_params.update(defaults)
        self.model_params.update(kwargs)

    def set_params(self, **kwargs):
        setter = self.setters.get(self.model_name)
        setter(**kwargs)
    
    def train_model(self):
        self._set_model()
        if self.model_params=={}:
            self.set_params() # if no params set, use defaults
        #wyowałaj grid_search
        # wywołaj train_forest

    def grid_search(self,score_type = "R2"):
        param_grid={
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        model = RandomForestRegressor(random_state=42)
        scorer = {
            "MAE": make_scorer(mean_absolute_error,greater_is_better=False),
            "MSE": make_scorer(mean_squared_error, greater_is_better=False),
            "R2": make_scorer(r2_score)        
        }
        if score_type not in scorer:
            raise ValueError(f"Niepoprawny typ metryki: {score_type}. Dostępne: {list(scorer.keys())}")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scorer,
            refit=score_type,
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        grid.fit(self.X_train,self.y_train)

        self.model = grid.best_estimator_

        cv_results = grid.cv_results_
        best_index = grid.best_index_
        outcomes = {
            "najlepsze_parametry": grid.best_params_,
            "sredni_r2": cv_results['mean_test_R2'][best_index],
            "sredni_mae": -cv_results['mean_test_MAE'][best_index],  
            "sredni_mse": -cv_results['mean_test_MSE'][best_index],  
        }

        print(outcomes)
        print(self.X_train.size)




        
         

    def train_forest(self):
        scoring = {
             "MAE": make_scorer(mean_absolute_error),
             "MSE": make_scorer(mean_squared_error),
             "RMSE": make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true,y_pred))),
             "R2": make_scorer(r2_score)
        }

        cv = cross_validate(
             RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs, n_estimators=self.n_estimators),
             self.X_train, self.y_train,
             cv=5,
             scoring=scoring
        )

        stats={
             "MAE": np.mean(cv["test_MAE"]),
             "MSE": np.mean(cv["test_MSE"]),
             "RMSE": np.mean(cv["test_RMSE"]),
             "R2": np.mean(cv["test_R2"]),
        }

        self.model = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs, n_estimators=self.n_estimators).fit(self.X_train, self.y_train)
        return stats
    
    def test_forest(self):
        prediction=self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, prediction)
        mse = mean_squared_error(self.y_test, prediction)
        r2 = r2_score(self.y_test, prediction) 
        rmse = np.sqrt(mse)
        
        stats = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        return stats
    
    def predict_forest(self, exa):
        if self.trans==None:
            exa = self.standardizer(exa)
            return self.model.predict(exa) 
        else:
            exa=self.trans.transform(exa)
            exa = self.standardizer(exa)
            return self.model.predict(exa) 
