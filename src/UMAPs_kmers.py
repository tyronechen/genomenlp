import numpy #for data interpretation
f_pos = open('Positive.w2v','r')
f_neg = open('Negative.w2v','r')
fcontent_pos = f_pos.read()
fcontent_neg = f_neg.read()
lis_pos = [x.split() for x in fcontent_pos.split('\n')[1:-1]]
lis1_pos = [[float(x) for x in y[1:]] for y in lis_pos]
lis_neg  = [x.split() for x in fcontent_neg.split('\n')[1:-1]]
lis1_neg = [[float(x) for x in y[1:]] for y in lis_neg]
l_pos = [x+[1] for x in lis1_pos]
l_neg = [x+[0] for x in lis1_neg]
l_whole = l_pos+l_neg 
dataset = numpy.array([numpy.array(x) for x in l_whole])
dataset.shape
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
X = dataset[:,0:-1]
Y = dataset[:,-1]
import umap # pip3 install umap-learn
manifold = umap.UMAP().fit(X, Y)
X_reduced = manifold.transform(X)
X_reduced.shape
import matplotlib.pyplot as plt
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, s=0.5);
plt.show()
from sklearn.preprocessing import QuantileTransformer
pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
X = pipe.fit_transform(X.copy())
manifold = umap.UMAP().fit(X, Y)
X_reduced_2 = manifold.transform(X)
plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1], c=Y, s=0.5);
plt.show()
