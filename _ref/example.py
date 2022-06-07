#!/usr/bin/python
import json
import re
# import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn import model_selection, feature_extraction, feature_selection
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and
    ## characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    ## Tokenize (convert from string to list)
    lst_text = text.split()    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
    ## back to string from list
    text = " ".join(lst_text)
    return text

def main():
  lst_dics = []
  # kaggle news category dataset
  print("###################### EXAMPLE INPUT DATA")
  print("Get the data here: https://www.kaggle.com/rmisra/news-category-dataset\n")
  with open('data.json', mode='r', errors='ignore') as json_file:
      for dic in json_file:
          lst_dics.append( json.loads(dic) )
  print(lst_dics[0])
  ## create dtf
  dtf = pd.DataFrame(lst_dics)
  # FIXME: changing this to a 2-case classification breaks feature selection! eg remove 'TECH'
  dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','POLITICS',]) ][["category","headline"]]
  dtf = dtf.rename(columns={"category":"y", "headline":"text"})
  print(dtf.sample(5))
  # fig, ax = plt.subplots()
  # fig.suptitle("y", fontsize=12)
  # dtf["y"].reset_index().groupby("y").count().sort_values(by="index").plot(
  #     kind="barh", legend=False, ax=ax
  #     ).grid(axis='x')
  # # plt.show()
  lst_stopwords = nltk.corpus.stopwords.words("english")
  # print(lst_stopwords)
  print("###################### EXAMPLE CLEAN DATA")
  dtf["text_clean"] = dtf["text"].apply(lambda x: utils_preprocess_text(
      x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords
  ))
  print(dtf.head())
  ## split dataset
  dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
  y_train = dtf_train["y"].values
  y_test = dtf_test["y"].values
  ## Count (classic BoW)
  vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))
  ## Tf-Idf (advanced variant of BoW)
  vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))
  corpus = dtf_train["text_clean"]
  vectorizer.fit(corpus)
  X_train = vectorizer.transform(corpus)
  dic_vocabulary = vectorizer.vocabulary_
  # sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')
  # plt.show()
  word = "new york"
  print(dic_vocabulary[word])
  y = dtf_train["y"]
  X_names = vectorizer.get_feature_names()
  p_value_limit = 0.95
  dtf_features = pd.DataFrame()
  for cat in np.unique(y):
      print(cat)
      chi2, p = feature_selection.chi2(X_train, y==cat)
      print(p)
      die
      dtf_features = dtf_features.append(pd.DataFrame(
                     {"feature":X_names, "score":1-p, "y":cat}
                     ))
      dtf_features = dtf_features.sort_values(["y","score"], ascending=[True,False])
      dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
      # print(dtf_features)
  X_names = dtf_features["feature"].unique().tolist()
  # print(X_names)
  print("###################### SELECTED FEATURES")
  # print("Dataframe indices unique is", dtf_features.index.is_unique)
  # print(dtf_features[dtf_features.index.duplicated(keep=False)])
  print("Dropping a class in line 43 makes selected features identical")
  for cat in np.unique(y):
     print("# {}:".format(cat))
     # print(dtf_features)
     # print("mask")
     # print(dtf_features["y"]==cat)
     # print("selected features")
     x = dtf_features[dtf_features["y"]==cat]
     print("  . selected features:", len(x))
     print("  . top features:", ",".join(x["feature"].values[:10]))
     print(" ")

if __name__ == '__main__':
    main()
