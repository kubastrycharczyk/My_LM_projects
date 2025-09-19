# Earthquake Prediction with Random Forest


---
**RUN 5.** Scores from cross-validation with polynomial features of degree 3 and proper order of standarization.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.42237241749183135    | 
| MSE     | 0.29907733354802096    | 
| RMSE     | 0.5466368229391567    | 
| R2     | 0.4918076973810413     | 





---
**RUN 4.** Scores from cross-validation with polynomial features of degree 2 and proper order of standarization.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.4225061196198438    | 
| MSE     | 0.29910073333451104    | 
| RMSE     | 0.546569101287967    | 
| R2     | 0.4919120261877922     | 





---
**RUN 3.** Scores from cross-validation with polynomial features of degree 2 but with wrong order of standarization.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.42719478179299486    | 
| MSE     | 0.3104968964071049    | 
| RMSE     | 0.5570535957879954    | 
| R2     | 0.46410815080059525     | 






---

**RUN 2.** Scores from cross-validation with grid search implementation.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.42845563898425976    | 
| MSE     | 0.30704408286457296    | 
| RMSE     | 0.553910595079019    | 
| R2     | 0.4782307434524992     | 



**Next step:** Polynomial features.   

---
**RUN 1.** Scores from cross-validation achieved after first training run.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.4263353552437346    | 
| MSE     | 0.3087935913713184    | 
| RMSE     | 0.5552154450490114    | 
| R2     | 0.4721765387710703     | 


**Analysis:** Errors of model are relativly low, that sugests that model well predicts values. Score of R2 seems to be insufficient explaining only 47% of variability.

**Next step:** Hyperparameter tuning.   

---

**Mistakes made:**
- Standarizing data before creating polynomial features


Earthquake data:
https://data-flair.training/blogs/earthquake-prediction-using-machine-learning/

Tectonic Plate Boundaries:
https://www.kaggle.com/datasets/cwthompson/tectonic-plate-boundaries?utm_source=chatgpt.com