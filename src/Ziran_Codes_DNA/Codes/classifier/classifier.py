!pip install shap
!pip install category_encoders
!pip install xgboost
!pip install --upgrade xgboost
!pip install hyperopt

import numpy as np
import pandas as pd
import re
import wandb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import category_encoders as ce
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from pprint import pprint
import matplotlib.pyplot as plt
# %matplotlib inline
# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

"""Relevant Functions for Classifiers; Hyperparameter Tuning; Feature Importance; Data Splitiing"""

#functions
def encode_labels(DataFrame, col_name):
    df_ce=DataFrame.copy()
    encoder=ce.OrdinalEncoder(cols=[col_name])
    df = encoder.fit_transform(df_ce)
    return df

def split_dataset(X, Y, train_ratio, test_ratio, validation_ratio):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    return x_train, y_train, x_test, y_test, x_val, y_val

def train_model(model,x_train, y_train):
   clf = model(n_estimators = 1000)
   # fit the training data into the model
   clf.fit(x_train, y_train)
   y_pred=clf.predict(x_test)
   y_probas=clf.predict_proba(x_test)
   return clf, y_pred, y_probas

def model_metrics(model, x_test, y_test, y_pred):
  # accuracy prediction
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy: %.2f%%" % (accuracy * 100.0))
  # classification report
  print("Classification report:\n")
  print(classification_report(y_test, y_pred))
  # confusion matrix
  conf=confusion_matrix(y_test, y_pred)
  print("Confusion matrix:\n", conf)
  # confusion matrix plot
  print("Confusion matrix plot:\n")
  plot_confusion_matrix(model, x_test, y_test)
  plt.show()

def shap_plot(model, x_test):
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(x_test)
   print('SHAP plot for Feature importance')
   shap.summary_plot(shap_values, x_test, plot_type="bar")

def feature_imp(model,feature_names, n_top_features):
  feats=np.array(feature_names)
  importances = model.feature_importances_
  indices = np.argsort(importances)[::-1]
  for f in range(50):
    print("%d. feature %d - %s : %f" % (f+1, indices[f], feats[f], importances[indices[f]]))
  plt.figure(figsize=(8,10))
  plt.barh(feats[indices][:n_top_features ], importances[indices][:n_top_features ])
  plt.xlabel("RF feature Importance ")

def token_freq_plot(token, top_features):
  freq=token.value_counts().nlargest(top_features)
  freq.plot(kind='bar', title='Token frequency distribution plot')

"""Combining Datasets and Labels from Positve/Negative Data"""

# data
# loaded the dna2vec word embedding file data
# importing the embeddings
pos = pd.read_csv("Positive.w2v", skiprows=1, sep=" ", index_col=0, header=None)
neg = pd.read_csv("Negative.w2v", skiprows=1, sep=" ", index_col=0, header=None)
# data preprocessing
# adding labels for natural and synthetic embeddings
pos['label']=1
neg['label']=0
# dataset
df=pd.concat([pos,neg])

#df.index.name='kmer'
#df.head()

df[:] = np.nan_to_num(df)
#Token freq plot
token_freq_plot(df.index, 30)

# split data into X and Y
# sequence embeddings
X=np.array(df.drop(['label'], axis=1))
# label of sequence embeddings
Y=np.array(df['label'])
# X and Y should have same length
print(X.shape)
print(Y.shape)

# split the dataset into train, test and validation set using sklearn
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
# train is now 70% of the entire data set
# test is now 15% of the initial data set
# validation is now 15% of the initial data set
x_train, y_train, x_test, y_test, x_val, y_val=split_dataset(X, Y, train_ratio, test_ratio, validation_ratio)
X = np.concatenate((X1, X0),axis = 0)
Y = np.concatenate((Y1,Y0), axis = 0)
print(X)
print(Y)

"""Training of Classifiers"""

# split the dataset into train, test and validation set using sklearn
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
# train is now 70% of the entire data set
# test is now 15% of the initial data set
# validation is now 15% of the initial data set
x_train, y_train, x_test, y_test, x_val, y_val=split_dataset(X, Y, train_ratio, test_ratio, validation_ratio)


"""XGBoost"""

print('XGBOOST BASE MODEL')
# training the model
model=XGBClassifier
param=100
xg_base, y_pred_xg, y_probas_xg=train_model(model, x_train, y_train)
# model metrics
model_metrics(xg_base, x_test, y_test, y_pred_xg)

# feature importance plot
feature_imp(xg_base,feature, 50)

# parameters cuurently used
print('Parameters currently in use:')
pprint(xg_base.get_params())



"""RandomForest"""

print('RANDOMFOREST BASE MODEL')
# training the model
model=RandomForestClassifier
param=100
rf_base, y_pred_rf, y_probas_rf=train_model(model, x_train, y_train)
# model metrics
model_metrics(rf_base, x_test, y_test, y_pred_rf)

# feature importance plot
feature_imp(rf_base,feature, 50)

# parameters cuurently used
print('Parameters currently in use:')
pprint(rf_base.get_params())



#parameters and scoring
param_xg = {
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [ 0.3, 0.5 , 0.8 ],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [0, 0.5, 1, 5],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [0, 0.5, 1, 5]
    }

param_rf = {'n_estimators': [100],
               'max_features': ['auto','sqrt'],
               'max_depth': [1,2,3],
               'min_samples_split': [2,3],
               'min_samples_leaf': [1,2],
               'bootstrap': [True, False]}
               
scoring = ['recall']

print('GRID SEARCH HYPERPARAMETER TUNING')

#Grid Search
def grid_search(model, param, scoring):
    estimator = model
    clf_grid=GridSearchCV(estimator=estimator, param_grid=param, scoring = scoring , refit = 'recall' , cv = 3, verbose=2, n_jobs = 4)
    # fit the training data
    clf_grid.fit(x_train,y_train)
    # Best hyperparameter values
    print('Best parameter values:')
    print(clf_grid.best_params_)
    # predicted values from the grid search model
    clf_g=clf_grid.best_estimator_
    y_pred=clf_g.predict(x_test)
    y_probas = clf_g.predict_proba(x_test)
    return clf_g, y_pred, y_probas

print('XGBOOST GRID SEARCH MODEL')

xg_grid, y_pred_xgg, y_probas_xgg=grid_search(XGBClassifier, param_xg, scoring)
# model metrics
model_metrics(xg_grid, x_test, y_test, y_pred_xgg)
# SHAP Plot
shap_plot(xg_grid, x_test)

# parameters cuurently used
print('Parameters currently in use:')
pprint(xg_grid.get_params())


print('RANDOMFOREST GRID SEARCH MODEL')


# Grid Search model
rf_grid, y_pred_rfg, yg_probas_rfg = grid_search(rf_base, param_rf, scoring)
# model metrics
model_metrics(rf_grid, x_test, y_test, y_pred_rfg)
# SHAP Plot
shap_plot(rf_grid, x_test)

# parameters cuurently used
print('Parameters currently in use:')
pprint(rf_grid.get_params())


print('RANDOM SEARCH HYPERPARAMETER TUNING')
#RANDOM SEARCH
def random_search(model, param, scoring):
    estimator = model
    clf_ran=RandomizedSearchCV(estimator=estimator, param_distributions = param, scoring = scoring, refit = 'recall', cv = 3, verbose=2, n_jobs = 4)
    # fit the training data
    clf_ran.fit(x_train,y_train)
    # Best hyperparameter values
    print('Best parameter values:')
    print(clf_ran.best_params_)
    # predicted values from the random search model
    clf_r=clf_ran.best_estimator_
    y_pred=clf_r.predict(x_test)
    y_probas = clf_r.predict_proba(x_test)
    return clf_r, y_pred, y_probas

print('XGBoost RANDOM SEARCH MODEL')
xg_random, y_pred_xgr, y_probas_xgr=random_search(XGBClassifier, param_xg, scoring)
# model metrics
model_metrics(xg_random, x_test, y_test, y_pred_xgr)
# SHAP Plot
shap_plot(xg_random, x_test)
# parameters cuurently used
print('Parameters currently in use:')
pprint(xg_random.get_params())


print('RANDOMFOREST GRID SEARCH MODEL')
# Random Search model
rf_random, y_pred_rfr, y_probas_rfr = random_search(rf_base, param_rf, scoring)
# model metrics
model_metrics(rf_grid, x_test, y_test, y_pred_rfr)
# SHAP Plot
shap_plot(rf_grid, x_test)

# parameters cuurently used
print('Parameters currently in use:')
pprint(rf_grid.get_params())


print('BAYESIAN OPTIMISATION')

scoring_ = 'recall''

#BAYESIAN OPTIMISATION
def bay_opt(model, param, scoring):
  # we need to define space from param grid
  # we are using choice-categorical variable distribution type to define space
  def space_(param):
    space = {key : hp.choice(key, param[key]) for key in param}
    return space

  #given below is an example of how space corresponds to param
  #param = {"learning_rate": [0.0001,0.001, 0.01, 0.1, 1] , "max_depth": range(3,21,3), "gamma": [i/10.0 for i in range(0,5)],"colsample_bytree": [i/10.0 for i in range(3,10)],"reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],"reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]}
  #space = {'learning_rate': hp.choice('learning_rate', [0.0001,0.001, 0.01, 0.1, 1]),'max_depth' : hp.choice('max_depth', range(3,21,3)),'gamma' : hp.choice('gamma', [i/10.0 for i in range(0,5)]),'colsample_bytree' : hp.choice('colsample_bytree', [i/10.0 for i in range(3,10)]),'reg_alpha' : hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),'reg_lambda' : hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100])}

  # Set up the k-fold cross-validation
  kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

  # Objective function
  # detach cross validation from sweep
  def objective(params):
    estimator = model(seed=0,**params)
    scores = cross_val_score(estimator, x_train, y_train, cv=kfold, scoring = scoring, n_jobs=-1)
    # Extract the best score
    best_score = max(scores)
    # Loss must be minimized
    loss = - best_score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
  # Trials to track progress
  bayes_trials = Trials()
  # Optimize
  best = fmin(fn = objective, space = space_(param), algo = tpe.suggest, max_evals = 48, trials = bayes_trials)
  # the index of the best parameters (best)
  # the values of the best parameters
  param_b = space_eval(space_(param), best)

  clf_bo = model(**param_b).fit(x_train, y_train)
  y_pred=clf_bo.predict(x_test)
  y_probas = clf_bo.predict_proba(x_test)
  return clf_bo, y_pred, y_probas


print('XGBoost BAYESIAN OPTIMISATION MODEL')
xg_bo, y_pred_xgo, y_probas_xgo = bay_opt(XGBClassifier, param_xg, scoring_)
# model metrics
model_metrics(xg_bo, x_test, y_test, y_pred_xgo)
# SHAP Plot
shap_plot(xg_bo, x_test)
# parameters cuurently used
print('Parameters currently in use:')
pprint(xg_bo.get_params())




print('RANDOMFOREST BAYESIAN OPTIMISATION MODEL')

rf_bo, y_pred_rfo, y_probas_rfo = bay_opt(RandomForestClassifier, param_rf, scoring_)
# model metrics
model_metrics(rf_bo, x_test, y_test, y_pred_rfo)
# SHAP Plot
shap_plot(rf_bo, x_test)
# parameters cuurently used
print('Parameters currently in use:')
pprint(rf_bo.get_params())








