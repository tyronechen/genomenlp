import wandb
wandb.login()
sweep_config = {
    "method": "random", # try grid or random
    "metric": {
      "name": "accuracy",
      "goal": "maximize"   
    },
    "parameters": {
        "booster": {
            "values": ["gbtree","gblinear"]
        },
        "max_depth": {
            "values": [3, 6, 9, 12]
        },
        "learning_rate": {
            "values": [0.1, 0.05, 0.2]
        },
        "subsample": {
            "values": [1, 0.5, 0.3]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")

# XGBoost model 
## Imported and downloaded the necessary modules for running XGBoost
import numpy 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
def train():
  config_defaults = {
    "booster": "gbtree",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1,
    "seed": 117,
    "test_size": 0.33,
  }

  wandb.init(config=config_defaults)  # defaults are over-ridden during the sweep
  config = wandb.config

  # load data
  f_pos = open('Positive.w2v','r') # opened the word embeddings(positive data)file trained on natural sequences from dna2vec saved on desktop 
  f_neg = open('Negative.w2v','r') # opened the word embeddings file trained on synthetic sequences(negative data)from dna2vec saved on desktop 
  fcontent_pos = f_pos.read() # read content on positive data
  fcontent_neg = f_neg.read() # read content on negative data
  lis_pos = [x.split() for x in fcontent_pos.split('\n')[1:-1]] 
  lis1_pos = [[float(x) for x in y[1:]] for y in lis_pos] 
  lis_neg  = [x.split() for x in fcontent_neg.split('\n')[1:-1]] # # took content from negative data
  lis1_neg = [[float(x) for x in y[1:]] for y in lis_neg]
  l_pos = [x+[1] for x in lis1_pos] # labelled natural sequence embeddings as 1
  l_neg = [x+[0] for x in lis1_neg] # labelled synthetic sequence embeddings as 0
  l_whole = l_pos+l_neg # merged both list containing positive sequence embeddings and negative
  dataset = numpy.array([numpy.array(x) for x in l_whole]) # converted the dataset into arrays for XGBoost implememtation

  # split data into X and Y
  X = dataset[:,0:-1] # X is sequence embeddings which needs to be classified
  Y = dataset[:,-1] # Y is label of sequence embeddings
  
  # split data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      test_size=config.test_size,
                                                      random_state=config.seed)

  # fit model on train
  model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=config.subsample)
  model.fit(X_train, y_train)

  # make predictions on test
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]

  # evaluate predictions
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy: {int(accuracy * 100.)}%")
  wandb.log({"accuracy": accuracy})
  
wandb.agent(sweep_id, train, count=25)
