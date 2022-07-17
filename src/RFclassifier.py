# -*- coding: utf-8 -*-

# RF classification model
# improves output by ignoring warnings
import warnings
warnings.filterwarnings('ignore')
# import all the required modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

# data
# loaded the dna2vec word embedding file data
# opened and read the respective files
f_pos=open('Positive.w2v','r')
f_neg=open('Negative.w2v','r')
file_p=f_pos.read()
file_n=f_neg.read()
# took file content as a list of sequences
# seperated by newline according to the indexing
lis_p=[x.split() for x in file_p.split('\n')[1:-1]]
lis_n=[x.split() for x in file_n.split('\n')[1:-1]]
# converted the sequence values(string) into numerical values(float) 
list_p=[[float(x) for x in y[1:]] for y in lis_p]
list_n=[[float(x) for x in y[1:]] for y in lis_n]
# labelled natural sequence embeddings as 1
l_pos=[x+[1] for x in list_p]
# labelled synthetic sequence embeddings as 0
l_neg=[x+[0] for x in list_n]
# merged both the lists together
l_whole = l_pos+l_neg
# converted the list to arrray for model implementation
dataset = np.array([np.array(x) for x in l_whole])

# split data into X and Y
# sequence embeddings
X = dataset[:,:-1]
# label of sequence embeddings
Y = dataset[:,-1]

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
