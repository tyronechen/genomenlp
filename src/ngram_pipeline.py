# import the required libraries
# !pip install fastaparser, version= 1.1.1
import pandas as pd
import numpy as np
import fastaparser
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from yellowbrick.text import FreqDistVisualizer
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_squared_error
from matplotlib.legend_handler import HandlerLine2D
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval
# Relevant functions
def Kmers_funct(seq, k_low, k_high):
   return [seq[x:x+random.randint(k_low,k_high)].lower() for x in range(len(seq)-k_low + 1)]

def spacings(seq,k_low,k_high):
  words = Kmers_funct(seq, k_low, k_high)
  joined_sentence = ' '.join(words)
  return joined_sentence

def loadsequence(sequence_file, k_low, k_high):
  with open(sequence_file) as fasta_file:
        parser = fastaparser.Reader(fasta_file, parse_method='quick')
        corpus = []
        for seq in parser:
          s = seq.sequence
          corpus.append(spacings(s, k_low, k_high))
  return corpus

def add_labels(sequence_file, k_low, k_high, label):
  file=loadsequence(sequence_file, k_low, k_high)
  file=pd.DataFrame(file)
  file.columns=['kmer']
  file['labels']=label
  return file

def countvec(corpus, labels, n_gram_from, n_gram_to, max_features):
  cv = CountVectorizer(ngram_range=(n_gram_from, n_gram_to), max_features=max_features)
  X=cv.fit_transform(corpus).toarray()
  feature=cv.get_feature_names_out()
  Y=np.array(labels)
  return X, Y, feature

def split_dataset(X, Y, train_ratio, test_ratio, validation_ratio):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    return x_train, y_train, x_test, y_test, x_val, y_val

def train_model(model, param, x_train, y_train, x_test):
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

def feature_imp(model,feature_names, n_top_features):
  feats=np.array(feature_names)
  importances = model.feature_importances_
  indices = np.argsort(importances)[::-1]
  plt.figure(figsize=(8,10))
  plt.barh(feats[indices][:n_top_features ], importances[indices][:n_top_features ])
  plt.xlabel("RF feature Importance ")

def grid_search(model, param, scoring, x_train, y_train, x_test):
    clf_grid=GridSearchCV(estimator=model, param_grid=param, scoring = scoring , refit = 'recall' , cv = 3, verbose=2, n_jobs = 4)
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

def random_search(model, param, scoring, x_train, y_train, x_test):
    clf_ran=RandomizedSearchCV(estimator=model, param_distributions = param, scoring = scoring, refit = 'recall', cv = 3, verbose=2, n_jobs = 4)
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

def bay_opt(model, param, scoring, x_train, y_train, x_test):
  # we need to define space from param grid
  # we are using choice-categorical variable distribution type to define space
  def space_(param):
    space = {key : hp.choice(key, param[key]) for key in param}
    return space
  # Set up the k-fold cross-validation
  kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

  # Objective function
  # detach cross validation from sweep
  def objective(params):
    estimator = model(**params)
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
   
def token_freq_plot(feature, X):
    visualizer = FreqDistVisualizer(features=feature, orient='v')
    visualizer.fit(X)
    visualizer.show()

def ROC_curve(model, x_test, y_test):
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax, alpha=0.8)
    plt.show()

def learning_curve(list_nb_trees, x_train, y_train, x_test, y_test):
    train_results = []
    test_results = []
    for nb_trees in list_nb_trees:
        rf = RandomForestClassifier(n_estimators=nb_trees)
        rf.fit(x_train, y_train)

        train_results.append(mean_squared_error(y_train, rf.predict(x_train)))
        test_results.append(mean_squared_error(y_test, rf.predict(x_test)))

    line1, = plt.plot(list_nb_trees, train_results, color="r", label="Training Score")
    line2, = plt.plot(list_nb_trees, test_results, color="g", label="Testing Score")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('MSE')
    plt.xlabel('n_estimators')
    plt.show()

def main():
        sequence_file1='Human_promoter.fa'
        sequence_file2='Human_promoter_synthetic.fa'
        k_low=5
        k_high=5
        label_1=0
        label_2=1
        n_gram_from=4
        n_gram_to=4
        max_features=1000
        train_ratio = 0.70
        validation_ratio = 0.15
        test_ratio = 0.15
        param=100
        model=RandomForestClassifier
        list_nb_trees=[100, 200, 300, 400, 500]

        pos=add_labels(sequence_file1, k_low, k_high, label_1)
        neg=add_labels(sequence_file2, k_low, k_high, label_2)
        dna=pd.concat([pos,neg]) 
        #print(dna.head())
        corpus=list(dna.kmer)
        X, Y, Feature_names=countvec(corpus, dna.labels, n_gram_from, n_gram_to, max_features)
        print("Total data iterms:",X.shape)
        print("Total data labels", Y.shape)
        token_freq_plot(Feature_names, X)
        # split the dataset into train, test and validation set using sklearn
        # train is now 70% of the entire data set
        # test is now 15% of the initial data set
        # validation is now 15% of the initial data set
        x_train, y_train, x_test, y_test, x_val, y_val=split_dataset(X, Y, train_ratio, test_ratio, validation_ratio) 
        print("Training data:",x_train.shape)
        print("Training data labels:",y_train.shape)
        print("Test data:",x_test.shape)
        print("Test data labels:",y_test.shape)
        print("Validation data:",x_val.shape)
        print("Validation data labels:",y_val.shape)
        print('RF BASE MODEL')
        ## training the model
        rf_base, y_pred, y_probas=train_model(model, param, x_train, y_train, x_test)
        # model metrics
        model_metrics(rf_base, x_test, y_test, y_pred)
        print("Feature Importance Plot:\n")
        feature_imp(rf_base, Feature_names, 20)
        print('ROC curve plot:\n')
        ROC_curve(rf_base, x_test, y_test)
        print('Learning curve:\n')
        learning_curve(list_nb_trees, x_train, y_train, x_test, y_test)
        # parameters currently used
        print('Parameters currently in use:')
        pprint(rf_base.get_params())
        # Setting range of parameters
        scoring = ['recall']
        param = {'n_estimators': [50, 100, 150, 200],
                      'max_features': ['auto', 'sqrt'],
                      'max_depth': [10, 20],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [2, 3],
                      'bootstrap': [True, False]}
        print('Range of parameters used for hyperparameter tuning:')
        pprint(param)
        #Hyperparameter tuning using GridsearchCV
        print("\nGRID SEARCH MODEL")
        # Grid Search model
        rf_grid, y_pred1, y_probas1=grid_search(rf_base, param, scoring, x_train, y_train, x_test)
        # model metrics
        model_metrics(rf_grid, x_test, y_test, y_pred1)
        print("Feature Importance Plot:\n")
        feature_imp(rf_grid, Feature_names, 20)
        print('ROC curve plot:\n')
        ROC_curve(rf_grid, x_test, y_test)
        print("\nRANDOM SEARCH MODEL")
        # Random search model
        rf_ran, y_pred2, y_probas2=random_search(rf_base, param, scoring, x_train, y_train, x_test)
        # model metrics
        model_metrics(rf_ran, x_test, y_test, y_pred2)
        print("Feature Importance Plot:\n")
        feature_imp(rf_ran, Feature_names, 20)
        print('ROC curve plot:\n')
        ROC_curve(rf_ran, x_test, y_test)
        print('BAYESIAN OPTIMISATION MODEL')
        scoring_rf='recall'
        rf_bayes, y_pred3, y_probas3 = bay_opt(RandomForestClassifier, param, scoring_rf, x_train, y_train, x_test)
        # model metrics
        model_metrics(rf_bayes, x_test, y_test, y_pred3)
        print("Feature Importance Plot:\n")
        feature_imp(rf_bayes, Feature_names, 20)
        print('ROC curve plot:\n')
        ROC_curve(rf_bayes, x_test, y_test)

if __name__ == "__main__":
    main()
