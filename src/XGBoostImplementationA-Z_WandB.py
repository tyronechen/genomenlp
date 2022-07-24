###### Step 1: Install And Import Libraries
import wandb
# Data processing
import numpy
# Model and performance evaluation
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import mean_squared_error

# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval # pip3 install hyperopt

# initialize wandb run
wandb.init(project="xgboost")

###### Step 2: Read In Data
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

# Check the number of records in training and testing dataset.
print(f'The training dataset has {len(X_train)} records.')
print(f'The testing dataset has {len(X_test)} records.')

# Initiate XGBoost Classifier
xgboost = XGBClassifier()
# Print default setting
xgboost.get_params()

xgboost.fit(X_train,y_train,callbacks=[wandb.xgboost.WandbCallback()])

preds = xgboost.predict(X_test)

rmse = numpy.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
wandb.log({"RMSE": rmse})


###### Step 4: Grid Search for XGBoost
# Define the search space
param_grid = {
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [ 0.3, 0.5 , 0.8 ],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [0, 0.5, 1, 5],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [0, 0.5, 1, 5]
    }
    
# Set up score
scoring = ['recall']
# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# Define grid search
grid_search = GridSearchCV(estimator=xgboost,
                           param_grid=param_grid,
                           scoring=scoring,
                           refit='recall',
                           n_jobs=-1,
                           cv=kfold,
                           verbose=0)

grid_search.fit(X_train,y_train,callbacks=[wandb.xgboost.WandbCallback()])

preds_gs = grid_search.predict(X_test)

rmse_gs = numpy.sqrt(mean_squared_error(y_test, preds_gs))
print("RMSE: %f" % (rmse_gs))
wandb.log({"RMSE": rmse_gs})



###### Step 4: Random Search for XGBoost
# Define the search space
param_grid = {
    # Learning rate shrinks the weights to make the boosting process more conservative
    "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
    # Maximum depth of the tree, increasing it increases the model complexity.
    "max_depth": range(3,21,3),
    # Gamma specifies the minimum loss reduction required to make a split.
    "gamma": [i/10.0 for i in range(0,5)],
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [i/10.0 for i in range(3,10)],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]}
# Set up score
scoring = ['recall']
# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# Define random search
random_search = RandomizedSearchCV(estimator=xgboost,
                           param_distributions=param_grid,
                           n_iter=48,
                           scoring=scoring,
                           refit='recall',
                           n_jobs=-1,
                           cv=kfold,
                           verbose=0)
                           
random_search.fit(X_train,y_train,callbacks=[wandb.xgboost.WandbCallback()])

preds_rs = random_search.predict(X_test)

rmse_rs = numpy.sqrt(mean_squared_error(y_test, preds_rs))
print("RMSE: %f" % (rmse_rs))
wandb.log({"RMSE": rmse_rs})


###### Step 5: Bayesian Optimization For XGBoost
# Space
space = {
    'learning_rate': hp.choice('learning_rate', [0.0001,0.001, 0.01, 0.1, 1]),
    'max_depth' : hp.choice('max_depth', range(3,21,3)),
    'gamma' : hp.choice('gamma', [i/10.0 for i in range(0,5)]),
    'colsample_bytree' : hp.choice('colsample_bytree', [i/10.0 for i in range(3,10)]),
    'reg_alpha' : hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),
    'reg_lambda' : hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100])
}
# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# Objective function
def objective(params):
    
    xgboost = XGBClassifier(seed=0, **params)
    scores = cross_val_score(xgboost, X_train, y_train, cv=kfold, scoring='recall', n_jobs=-1)
    # Extract the best score
    best_score = max(scores)
    # Loss must be minimized
    loss = - best_score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
# Trials to track progress
bayes_trials = Trials()
# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 48, trials = bayes_trials)
# Print the index of the best parameters
print(best)
# Print the values of the best parameters
print(space_eval(space, best))
# Train model using the best parameters
xgboost_bo = XGBClassifier(seed=0,
                           colsample_bytree=0.4,
                           gamma=0.2,
                           learning_rate=1,
                           max_depth=12,
                           reg_alpha=1e-05,
                           reg_lambda=1
                           ).fit(X_train,y_train,callbacks=[wandb.xgboost.WandbCallback()])

bayesian_opt_predict = xgboost_bo.predict(X_test)
rmse_ho = numpy.sqrt(mean_squared_error(y_test, bayesian_opt_predict))
print("RMSE: %f" % (rmse_ho))
wandb.log({"RMSE": rmse_ho})


                          
