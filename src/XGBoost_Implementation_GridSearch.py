# XGBoost model
import xgboost
import numpy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV #Hyperparamter Tuning

# load data
f_pos = open('Positive.w2v','r')
f_neg = open('Negative.w2v','r')
fcontent_pos = f_pos.read()
fcontent_neg = f_neg.read()
lis_pos = [x.split() for x in fcontent_pos.split('\n')[1:50]]
lis1_pos = [[float(x) for x in y[1:]] for y in lis_pos]
lis_neg  = [x.split() for x in fcontent_neg.split('\n')[1:50]]
lis1_neg = [[float(x) for x in y[1:]] for y in lis_neg]
l_pos = [x+[0] for x in lis1_pos]
l_neg = [x+[1] for x in lis1_neg]
l_whole = l_pos+l_neg
dataset = numpy.array([numpy.array(x) for x in l_whole])

# split data into X and Y
X = dataset[:,0:-1]
Y = dataset[:,-1]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#classifier
classifier = xgboost.XGBClassifier()

#tunable paramters and range of values
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

#grid serach object
grid_search = GridSearchCV(
    classifier,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)
#training
grid_search.fit(X, Y)

#best estimator
grid_search.best_estimator_
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
