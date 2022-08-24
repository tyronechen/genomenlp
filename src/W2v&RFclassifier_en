# -*- coding: utf-8 -*-

#import all the required modules
import pandas as pd

import re

import os

import gensim

from matplotlib import pyplot as plt
%matplotlib inline

import numpy as np

import nltk
nltk.download('punkt')

!pip install category_encoders
import category_encoders as ce

# improves output by ignoring warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score

from pprint import pprint

# Bayesian optimization
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

!pip install wandb
import wandb

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text import FreqDistVisualizer




# read the dataset
df=pd.read_csv('fake_or_real_news.csv')
#df.head()

# text is the target variable
# Cleaning the text of punctuation and special characters
# and added the clean text as the column 'clean' in the dataframe
clean_txt = []
for w in range(len(df.text)):
   desc = df['text'][w].lower()

   #remove punctuation
   desc = re.sub('[^a-zA-Z]', ' ', desc)

   #remove tags
   desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)

   #remove digits and special chars
   desc=re.sub("(\\d|\\W)+"," ",desc)
   clean_txt.append(desc)

df['clean'] = clean_txt
#df.head()

# Creating the corpus
corpus = []
for col in df.clean:
   word_list = col.split(" ")
   corpus.append(word_list)

#show first value
print(corpus[0:1])

# Word2vec
 class MySentences(object):
    """MySentences is a generator to produce a list of tokenized sentences 
    
    Takes a list of numpy arrays containing documents.
    
    Args:
        arrays: List of arrays, where each element in the array contains a document.
    """
    def __init__(self, *arrays):
        self.arrays = arrays
 
    def __iter__(self):
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document):
                    yield nltk.word_tokenize(sent)
def get_word2vec(sentences, location):
    """Returns trained word2vec
    
    Args:
        sentences: iterator for sentences
        
        location (str): Path to save/load word2vec
    """
    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model
    
    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model


 w2vec = get_word2vec(
    MySentences(
        df['clean'].values, 
    ),
    'w2vmodel'
)

class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


        
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
           ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)
mean_embedded = mean_embedding_vectorizer.fit_transform(df['clean'])

df['array']=list(mean_embedded)
df.head()

# encoding column label
df_ce=df.copy()
encoder=ce.OrdinalEncoder(cols=['label'])
df_en = encoder.fit_transform(df_ce)
df_en.head()

# RF Classifier
#split data into X and Y
X=np.array([np.array(x) for x in df_en['array']])
Y=np.array([np.array(x) for x in df_en['label']])

# split the data into train and test using sklearn
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
# test_size defines test data to be split from train data
# 1/3rd of dataset(training) is used as test dataset

# RF classifier model
# called with default values
classifier=RandomForestClassifier()

# fit the training data into the base model
classifier.fit(x_train, y_train)

# predicted values from the model
y_pred=classifier.predict(x_test)
y_probas=classifier.predict_proba(x_test)

# accuracy prediction
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# classification report
print("Classification report:\n")
print(classification_report(y_test, y_pred))

# confusion matrix
conf=confusion_matrix(y_test, y_pred)
print("Confusion matrix\n", conf)

# confusion matrix plot
print("Confusion matrix plot:\n")
plot_confusion_matrix(classifier, x_test, y_test)  

# Weights and biases plot
#RF BASE Model
wandb.init(project="RF classifier-english dataset", name="RF-base model")

# Feature importance
wandb.sklearn.plot_feature_importances(classifier)
# metrics summary
wandb.sklearn.plot_summary_metrics(classifier, x_train, y_train, x_test, y_test)
# precision recall
wandb.sklearn.plot_precision_recall(y_test, y_probas, labels=None)
# ROC curve
wandb.sklearn.plot_roc(y_test, y_probas, labels=None)
# Learning curve
wandb.sklearn.plot_learning_curve(classifier, x_train, y_train)
# class proportions
wandb.sklearn.plot_class_proportions(y_train, y_test, labels=None)
# calibration curve
wandb.sklearn.plot_calibration_curve(classifier, X, Y, 'RandomForestClassifier- Base model')
#confusion matrix
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=None)

# parameters cuurently used
print('Parameters currently in use:')
pprint(classifier.get_params())

# Hyperparameter tuning using GridsearchCV
# Setting range of parameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
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
# implemented grid search on RF classifier
classifier_grid=GridSearchCV(estimator=classifier, param_grid=param, cv = 3, verbose=2, n_jobs = 4)
# fit the training data
classifier_grid.fit(x_train,y_train)

# Best hyperparameter values
print('Best parameter values:')
print(classifier_grid.best_params_)

# predicted values from the grid search model
cl_g=classifier_grid.best_estimator_
pred=cl_g.predict(x_test)
y_probas = cl_g.predict_proba(x_test)

# accuracy prediction for grid search model
accuracy = accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#classification report
print("Classification report:\n")
print(classification_report(y_test, y_pred))

# confusion matrix
conf_g=confusion_matrix(y_test, pred)
print("Confusion matrix\n", conf_g)

# confusion matrix plot
print("Confusion matrix plot:\n")
plot_confusion_matrix(cl_g, x_test, y_test)  
plt.show()

# wandb plot
# Hyperparameter tuning 
# Grid search model
wandb.init(project="RF classifier-english dataset", name="RF-gridsearch model")
# Feature importance
wandb.sklearn.plot_feature_importances(cl_g)
# metrics summary
wandb.sklearn.plot_summary_metrics(cl_g, x_train, y_train, x_test, y_test)
# precision recall
wandb.sklearn.plot_precision_recall(y_test, y_probas, labels=None)
# ROC curve
wandb.sklearn.plot_roc(y_test, y_probas, labels=None)
# Learning curve
wandb.sklearn.plot_learning_curve(cl_g, x_train, y_train)
# class proportions
wandb.sklearn.plot_class_proportions(y_train, y_test, labels=None)
# calibration curve
wandb.sklearn.plot_calibration_curve(cl_g, X, Y, 'RandomForestClassifier-Grid search model')
# confusion matrix
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=None)

# Random search implementation
# Module for hyperparameter tuning
# Hyperparameter tuning using RandomizedsearchCV
# Setting range of parameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print("\nRANDOM SEARCH MODEL")
print('Range of parameters used for hyperparameter tuning:')
pprint(param)

# implemented grid search on RF classifier
classifier_random=RandomizedSearchCV(estimator = classifier, param_distributions = param, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# fit the training data in the randomized model
classifier_random.fit(x_train, y_train)

# Best hyperparameter values
print('Best parameter values:')
print(classifier_random.best_params_)

# predicted values from the random search model using best parameters
cl_r=classifier_random.best_estimator_
pred=cl_r.predict(x_test)
y_probas = cl_r.predict_proba(x_test)

# accuracy prediction for random search model
accuracy = accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#classification report
print("Classification report:\n")
print(classification_report(y_test, y_pred))

# confusion matrix
conf_r=confusion_matrix(y_test, pred)
print("Confusion matrix\n", conf_r)

# confusion matrix plot
print("Confusion matrix plot:\n")
plot_confusion_matrix(cl_r, x_test, y_test)  

# wandb plot
# Random search
wandb.init(project="RF classifier-english dataset", name="RF-random search model")
# Feature importance
wandb.sklearn.plot_feature_importances(cl_r)
# metrics summary
wandb.sklearn.plot_summary_metrics(cl_r, x_train, y_train, x_test, y_test)
# precision recall
wandb.sklearn.plot_precision_recall(y_test, y_probas, labels=None)
# ROC curve
wandb.sklearn.plot_roc(y_test, y_probas, labels=None)
# Learning curve
wandb.sklearn.plot_learning_curve(cl_r, x_train, y_train)
# class proportions
wandb.sklearn.plot_class_proportions(y_train, y_test, labels=None)
# calibration curve
wandb.sklearn.plot_calibration_curve(cl_r, X, Y, 'RandomForestClassifier-Random search model')
# confusion matrix
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=None)

# Bayesian optimization
print("\nBAYESIAN OPTIMIZATION")
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
    }
def objective(space):
   model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
   #5 times cross validation fives 5 accuracies=>mean of these accuracies will be considered
   accuracy = cross_val_score(model, x_train, y_train, cv = 5).mean()
   # We aim to maximize accuracy, therefore we return it as a negative value
   return {'loss': -accuracy, 'status': STATUS_OK }

from sklearn.model_selection import cross_val_score
trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)   
crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}
print("Best parameters:")
print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])
rf_bayesian=RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'], 
                                       max_features = feat[best['max_features']], 
                                       min_samples_leaf = best['min_samples_leaf'], 
                                       min_samples_split = best['min_samples_split'], 
                                       n_estimators = est[best['n_estimators']]).fit(x_train,y_train)
pred_b=rf_bayesian.predict(x_test)
y_probas = rf_bayesian.predict_proba(x_test)
# accuracy prediction
accuracy = accuracy_score(y_test, pred_b)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# classification report
print("Classification report:\n")
print(classification_report(y_test, pred_b))

# confusion matrix
conf=confusion_matrix(y_test, pred_b)
print("Confusion matrix:\n", conf)

# confusion matrix plot
print("Confusion matrix plot:\n")
plot_confusion_matrix(rf_bayesian, x_test, y_test)  
plt.show()

#wandb plot
wandb.init(project="RF classifier-english dataset", name="RF-Bayesian optimization model")
# Feature importance
wandb.sklearn.plot_feature_importances(model=rf_bayesian, title="RF-Bayesian model")
# metrics summary
wandb.sklearn.plot_summary_metrics(rf_bayesian, x_train, y_train, x_test, y_test)
# precision recall
wandb.sklearn.plot_precision_recall(y_test, y_probas, labels=None)
# ROC curve
wandb.sklearn.plot_roc(y_test, y_probas, labels=None)
# Learning curve
wandb.sklearn.plot_learning_curve(rf_bayesian, x_train, y_train)
# class proportions
wandb.sklearn.plot_class_proportions(y_train, y_test, labels=None)
# calibration curve
wandb.sklearn.plot_calibration_curve(rf_bayesian, X, Y, 'RandomForestClassifier-Bayesian optimization model')
# confusion matrix
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=None)

# Exporting metrics from a project in to a CSV file

api = wandb.Api()
entity, project = "tyagilab", "RF classifier-english dataset"  
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

runs_df.to_csv("project_en.csv")


# Token Frequency distribution plot- Freqdist
vectorizer = CountVectorizer()
docs       = vectorizer.fit_transform(df.clean)
features   = vectorizer.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(docs)
visualizer.show()
