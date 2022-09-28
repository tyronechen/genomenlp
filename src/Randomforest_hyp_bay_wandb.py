# -*- coding: utf-8 -*-

# RF classification model
# import the required modules
# improves output by ignoring warnings
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from pprint import pprint
import matplotlib.pyplot as plt
%matplotlib 
#wandb
#!pip install wandb
import wandb
#wandb.init('RF classifier')
#SHAP
#!pip install shap
import shap
# Bayesian optimization
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

#functions
def split_dataset(X, Y, train_ratio, test_ratio, validation_ratio):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    return x_train, y_train, x_test, y_test, x_val, y_val

def train_model(model, param, x_train, y_train):
   clf=model(param)
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

def grid_search(model, param):
    clf_grid=GridSearchCV(estimator=model, param_grid=param, cv = 3, verbose=2, n_jobs = 4)
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

def random_search(model, param):
    clf_ran=RandomizedSearchCV(estimator=model, param_distributions=param, cv = 3, verbose=2, n_jobs = 4)
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
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

print('RF BASE MODEL')
# training the model
model=RandomForestClassifier
param=1000
rf_base, y_pred, y_probas=train_model(model, param, x_train, y_train)
# model metrics
model_metrics(rf_base, x_test, y_test, y_pred)
# SHAP Plot
shap_plot(rf_base, x_test) 
# wandb plot
wandb.init(project="RF classifier", name="RF-base model")
# Visualize all classifier plots at once
wandb.sklearn.plot_classifier(rf_base, x_train, x_test, y_train, y_test, y_pred, y_probas, labels=None, model_name='BASE MODEL', feature_names=None)
#Feature imp plot
feature_imp(rf_base, df.index, 50)

# parameters cuurently used
print('Parameters currently in use:')
pprint(rf_base.get_params())

# Hyperparameter tuning using GridsearchCV
# Setting range of parameters
# Number of trees in random forest
n_estimators = [1000, 2000, 3000, 4000, 5000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [1, 2, 3, 4, 5]
# Minimum number of samples required to split a node
min_samples_split = [1, 2, 3]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print("\nGRID SEARCH MODEL")
print('Range of parameters used for hyperparameter tuning:')
pprint(param)

# Grid Search model
rf_grid, y_pred1, y_probas1=grid_search(model, param)
# model metrics
model_metrics(rf_grid, x_test, y_test, y_pred1)
# SHAP Plot
shap_plot(rf_grid, x_test) 
# wandb plot
wandb.init(project="RF classifier", name="RF-gridsearch model")
# Visualize all classifier plots at once
wandb.sklearn.plot_classifier(rf_grid, x_train, x_test, y_train, y_test, y_pred1, y_probas1, labels=None, model_name='Grid Search Model', feature_names=None)
#Feature imp plot
feature_imp(rf_grid, df.index, 50)

print("\nRANDOM SEARCH MODEL")
print('Range of parameters used for hyperparameter tuning:')
pprint(param)
# Random search model
rf_ran, y_pred2, y_probas2=random_search(rf_base, param)
# model metrics
model_metrics(rf_ran, x_test, y_test, y_pred2)
# SHAP Plot
shap_plot(rf_ran, x_test) 
# wandb plot
wandb.init(project="RF classifier", name="RF-random search model")
# Visualize all classifier plots at once
wandb.sklearn.plot_classifier(rf_ran, x_train, x_test, y_train, y_test, y_pred2, y_probas2, labels=None, model_name='Random Search Model', feature_names=None)
#Feature imp plot
feature_imp(rf_ran, df.index, 50)

# Bayesian
# Exporting metrics from a project in to a CSV file

api = wandb.Api()
entity, project = "tyagilab", "RF classifier"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")
