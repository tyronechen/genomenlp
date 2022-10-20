# Expected
1. Base model of XGB and RF
2. GridSearch of XGB and RF
3. RandomSearch of XGB and RF
4. Bayesian Optimisation of XGB and RF

# Obtained
Expected 
All functions written inside code ran successfully

# Arg
param_xg = parameters for running hyperparameter tuning on XGB
param_rf = parameters for running hyperparameter tuning on RF
scoring = parameter(statistical value) we want to maximise for training; Eg: scoring = ['recall'] for GridSearch and RandomSearch, scoring = 'recall' for Bayesian Optimisation
space = formed by param grids for bayesian optimisation


# IO
input file is 'Positive.w2v' and 'Negative.w2v' which can be found on below link:
https://drive.google.com/drive/folders/1WBh1ek_-i46sU412ycWtMSdA26LPnFxK?usp=sharing

output is decribed in expected for each of functions
​
# Environment
shap
category_encoders
xgboost
hyperopt
numpy
pandas
re
wandb
sklearn
