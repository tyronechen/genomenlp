# XGBoost model 
## Imported and downloaded the necessary modules for running XGBoost
import numpy 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from pprint import pprint

# load data
f_pos = open('Positive.w2v','r') # opened the word embeddings file trained on natural sequences(positive data) from dna2vec
f_neg = open('Negative.w2v','r') # opened the word embeddings file trained on synthetic sequences(negative data)from dna2vec 
fcontent_pos = f_pos.read() # read content on positive data
fcontent_neg = f_neg.read() # read content on negative data
lis_pos = [x.split() for x in fcontent_pos.split('\n')[1:-1]] # took content from positive data in form of list of sequence embeddings separated by line from second line to last line # excluded first line here because it is not desired output from dna2vec-it is just the matrix dimension of resulting embeddings 
lis1_pos = [[float(x) for x in y[1:]] for y in lis_pos] # converted the list elements to float(numerical values) from strting(default datatype when read from file)- here we had left out k-mer such as AAA since that is of no need- we only need embeddings(vector)-that is why we had included from elements first value i.e y[1:]
lis_neg  = [x.split() for x in fcontent_neg.split('\n')[1:-1]] # # took content from negative data
lis1_neg = [[float(x) for x in y[1:]] for y in lis_neg] # converted the list elements to float(numerical values) from strting(default datatype when read from file)- here we had left out k-mer such as AAA since that is of no need- we only need embeddings(vector)-that is why we had included from elements first value i.e y[1:]
l_pos = [x+[1] for x in lis1_pos] # labelled natural sequence embeddings as 1
l_neg = [x+[0] for x in lis1_neg] # labelled synthetic sequence embeddings as 0
l_whole = l_pos+l_neg # merged both list containing positive sequence embeddings and negative
dataset = numpy.array([numpy.array(x) for x in l_whole]) # converted the dataset into arrays for XGBoost implememtation

# split data into X and Y
X = dataset[:,0:-1] # X is sequence embeddings which needs to be classified
Y = dataset[:,-1] # Y is label of sequence embeddings
# split data into train and test sets
seed = 7 # random state is defined for making training less bias prone
test_size = 0.33 # test dataset is 1/3 of dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) # splitted dataset into training and testing data

# fit model no training data
model = XGBClassifier() # XGBoost classifier is called
model.fit(X_train, y_train) # model is trained on training dataset

# print prediction results on test data
predictions = model.predict(X_test) 
print(classification_report(y_test, predictions)) 

## This is the script common as our implementation of XGBoost which runs on default Hyperparameters

# Look at hyperparameters used by our current model
print('Parameters currently in use:')
pprint(model.get_params())

# Random Search
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

#defining range of hyperparamters to tune our model 
colsample_bytree = [0.3, 0.4, 0.5, 0.7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
learning_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
max_depth = range(2,10,1)
min_child_weight = range(1,10,2)
n_estimators = range(0, 200, 10)

param = {'n_estimators': n_estimators,
    'max_depth':max_depth,
    'min_child_weight': min_child_weight,
    'gamma':gamma,
    'colsample_bytree':colsample_bytree,
    'learning_rate':[0.05,0.10,0.15,0.20,0.25,0.30]
}

pprint(param) #print the hyperparamters  range that is going to be used for tuning

classifier_random= RandomizedSearchCV(estimator=model, param_grid=param, cv = 3, verbose=2, n_jobs = 4) #defined classifier grid 
#1.estimator: Pass the model instance for which you want to check the hyperparameters.
#2.params_grid: the dictionary object that holds the hyperparameters you want to try
#3.scoring: evaluation metric that you want to use, you can simply pass a valid string/ object of evaluation metric
#4.cv: number of cross-validation you have to try for each selected set of hyperparameters
#5.verbose: you can set it to 1 to get the animated print out while you fit the data to GridSearchCV, 2 will give epoch data
#6.n_jobs: number of processes you wish to run in parallel for this task if it is -1 it will use all available processes. 

classifier_random.fit(X_train,y_train) #trained GridSearchCV on training data
classifier_random.best_params_  # returns best hyperparameters

# print prediction results on test data
predictions = classifier_random.predict(X_test) 
print(classification_report(y_test, predictions)) 

















