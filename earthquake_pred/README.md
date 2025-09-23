# Earthquake Prediction with Random Forest


**RUN 5.** Scores from cross-validation with polynomial features of degree 1 and adding new feature of platue distance.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.40515047416526284    | 
| MSE     | 0.2782793132302996    | 
| R2     | 0.5272723683910844     | 

**Analysis:** Feature engineering had slightly improvment from the best score, improvment around 0.6% point in R2. 

**Next step:** Adding lightgbm to the project.


---
**RUN 4.** Scores from cross-validation with polynomial features of degree 3 and proper order of standarization.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.4102255488013915    | 
| MSE     | 0.2839492976532145    | 
| R2     | 0.517611637264747     | 

**Analysis:** Polynomial features of degree 3 gives minimal worse results than degree 2. 

**Next step:** Performing feature engineering, adding distance from tectonic platue and type of it.


---
**RUN 3.** Scores from cross-validation with polynomial features of degree 2 and proper order of standarization.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.40987615224550994    | 
| MSE     | 0.2836645332469261    | 
| R2     | 0.518226749996563     | 


**Analysis:** Polynomial features of degree 2 gives minimal worse results which is excepcted because forests are able to reflect non-linear schemes. 

**Next step:** Just in case, performing degree 3.

---

**RUN 2.** Scores from cross-validation with grid search implementation.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.4080624310856427    | 
| MSE     | 0.2822416754867173    | 
| R2     | 0.5205997973225613     | 

**Analysis:** Grid search implementation had significant impact on r2 score with improvment around 5% points. 

**Next step:** Polynomial features, the problem might be in small number of features in set that consists only of 3 predictors.   
---
**RUN 1.** Scores from cross-validation achieved after first training run.  


| Scoring type | Score |
|------------|------------|
| MAE    | 0.42845563898425965    | 
| MSE     | 0.30704408286457285    | 
| R2     | 0.47823074345249916     | 


**Analysis:** Errors of model are relativly low, that sugests that model well predicts values. Score of R2 seems to be insufficient explaining only 47% of variability.

**Next step:** Hyperparameter tuning.   

---

**Mistakes made:**
- Standarizing data before creating polynomial features


Earthquake data:
https://data-flair.training/blogs/earthquake-prediction-using-machine-learning/

Tectonic Plate Boundaries:
https://www.kaggle.com/datasets/cwthompson/tectonic-plate-boundaries?utm_source=chatgpt.com