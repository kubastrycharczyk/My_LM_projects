# Earthquake Prediction with Random Forest

{'najlepsze_parametry': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}, 'sredni_r2': np.float64(0.5110667007158745), 'sredni_mae': np.float64(0.40675711343383225), 'sredni_mse': np.float64(0.2832233679541909)}
19575
{'MAE': np.float64(0.42719478179299486),
 'MSE': np.float64(0.3104968964071049),
 'RMSE': np.float64(0.5570535957879954),
 'R2': np.float64(0.46410815080059525)}



{'najlepsze_parametry': {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}, 'sredni_r2': np.float64(0.5088833692307035), 'sredni_mae': np.float64(0.4086009610771182), 'sredni_mse': np.float64(0.28878114951401057)}


---
**RUN 1.** Scores from cross-validation achieved after first training run.  
|Date: |10.09.2025|
|------------|------------|

| Scoring type | Score |
|------------|------------|
| MAE    | 0.4263353552437346    | 
| MSE     | 0.3087935913713184    | 
| RMSE     | 0.5552154450490114    | 
| R2     | 0.4721765387710703     | 


**Analysis:** Errors of model are relativly low, that sugests that model well predicts values. Score of R2 seems to be insufficient explaining only 47% of variability.

**Next step:** Hyperparameter tuning.   

---