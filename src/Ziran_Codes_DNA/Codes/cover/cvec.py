from sklearn.feature_extraction.text import CountVectorizer
from preprocess_dna import loadsequence
from yellowbrick.text import FreqDistVisualizer

def cvec(sequence_file, k_low, k_high,label):
  cv = CountVectorizer(max_features=1000,ngram_range=(1,3))
  corpus = loadsequence(sequence_file, k_low, k_high)
  X = cv.fit_transform(corpus).toarray()
  Y = np.array([label for i in range(X.shape[0])])
  return [X,Y]

def cvec_n_topfeatures(sequence_file, k_low, k_high,n):
  cv = CountVectorizer(max_features=1000,ngram_range=(1,3))
  corpus = loadsequence(sequence_file, k_low, k_high)
  X = cv.fit_transform(corpus).toarray()
  return cv.get_feature_names_out()[:n]

def visualise_cvec_freq_dist(sequence_file, k_low, k_high):
  cv = CountVectorizer(max_features=1000,ngram_range=(1,3))
  corpus = loadsequence(sequence_file, k_low, k_high)
  X = cv.fit_transform(corpus).toarray()
  feature=cv.get_feature_names_out()
  visualizer = FreqDistVisualizer(features=feature, orient='v')
  visualizer.fit(X)
  visualizer.show()

#sequence_file = 'PromoterSet.fa'
#k_low = 3
#k_high = 5

# X and Y should have same length
#label = 1
#n=30
#print(cvec(sequence_file, k_low, k_high,label)[0][1:10])
#print(cvec(sequence_file, k_low, k_high,label)[1][1:10])
#print(cvec_n_topfeatures(sequence_file, k_low, k_high,n))
#visualise_cvec_freq_dist(sequence_file, k_low, k_high)

#sequence_file = 'PromoterSet.fa'
#k_low = 3
#k_high = 5



