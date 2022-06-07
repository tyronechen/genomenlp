#!/usr/bin/python
## for data
from joblib import parallel_backend, Parallel, delayed, dump, load
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sklearn
from sklearn import feature_extraction, feature_selection, model_selection, naive_bayes, pipeline, manifold, preprocessing## for explainer
from sklearn.ensemble import RandomForestClassifier
from lime import lime_text
import gensim
import gensim.downloader as gensim_api
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
import transformers

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

def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

def main():
    lst_dics = []
    # kaggle news category dataset
    with open('data.json', mode='r', errors='ignore') as json_file:
        for dic in json_file:
            lst_dics.append( json.loads(dic) )
    print(lst_dics[0])
    ## create dtf
    dtf = pd.DataFrame(lst_dics) #'TECH'
    dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','POLITICS',]) ][["category","headline"]]
    dtf = dtf.rename(columns={"category":"y", "headline":"text"})
    print(dtf.sample(5))
    fig, ax = plt.subplots()
    fig.suptitle("y", fontsize=12)
    dtf["y"].reset_index().groupby("y").count().sort_values(by="index").plot(
        kind="barh", legend=False, ax=ax
        ).grid(axis='x')
    # plt.show()
    lst_stopwords = nltk.corpus.stopwords.words("english")
    # print(lst_stopwords)
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
        chi2, p = feature_selection.chi2(X_train, y==cat)
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat}
                       ))
        dtf_features = dtf_features.sort_values(["y","score"], ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()
    for cat in np.unique(y):
       print("# {}:".format(cat))
       print("  . selected features:", len(dtf_features[dtf_features["y"]==cat]))
       print("  . top features:", ",".join(dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
       print(" ")
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_
    classifier = naive_bayes.MultinomialNB()
    classifier = RandomForestClassifier(n_estimators=500)
    with parallel_backend('threading', n_jobs=6):
        model = pipeline.Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])## train classifier
        model["classifier"].fit(X_train, y_train)## test
    X_test = dtf_test["text_clean"].values
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    ## Accuracy, Precision, Recall
    accuracy = sklearn.metrics.accuracy_score(y_test, predicted)
    auc = sklearn.metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    print("Accuracy:", round(accuracy,2))
    print("Auc:", round(auc,2))
    print("Detail:")
    print(sklearn.metrics.classification_report(y_test, predicted))
    ## Plot confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], sklearn.metrics.auc(fpr, tpr)))
    ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[1].plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], sklearn.metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    # plt.show()
    ## select observation
    i = 0
    txt_instance = dtf_test["text"].iloc[i]## check true value and predicted value
    print("True:", y_test[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))## show explanation
    print(y_train)
    explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
    print(explainer)
    print(txt_instance)
    print(model.predict_proba)
    explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=3)
    # explained.show_in_notebook(text=txt_instance, predict_proba=False)
    # plt.show()
    print(explained)
    explained.save_to_file("tmp1.html")
    # plt.show()

    ##### NEXT
    nlp = gensim_api.load("word2vec-google-news-300")
    corpus = dtf_train["text_clean"]
    ## create list of lists of unigrams
    lst_corpus = []
    for string in corpus:
       lst_words = string.split()
       lst_grams = [" ".join(lst_words[i:i+1])
                   for i in range(0, len(lst_words), 1)]
       lst_corpus.append(lst_grams)
    ## detect bigrams and trigrams
    bigrams_detector = gensim.models.phrases.Phrases(lst_corpus,
                     delimiter=" ", min_count=1, threshold=1)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
                     delimiter=" ", min_count=1, threshold=1)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    ## fit w2v
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=300,
            window=8, min_count=1, sg=1, epochs=30)
    word = "data"
    print(nlp.wv[word].shape)

    word = "data"

    fig = plt.figure()## word embedding
    tot_words = [word] + [tupla[0] for tupla in nlp.wv.most_similar(word, topn=20)]
    X = nlp.wv[tot_words]## pca to reduce dimensionality from 300 to 3
    pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
    X = pca.fit_transform(X)## create dtf
    dtf_ = pd.DataFrame(X, index=tot_words, columns=["x","y","z"])
    dtf_["input"] = 0
    dtf_["input"].iloc[0:1] = 1## plot 3d
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dtf_[dtf_["input"]==0]['x'],
               dtf_[dtf_["input"]==0]['y'],
               dtf_[dtf_["input"]==0]['z'], c="black")
    ax.scatter(dtf_[dtf_["input"]==1]['x'],
               dtf_[dtf_["input"]==1]['y'],
               dtf_[dtf_["input"]==1]['z'], c="red")
    ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[],
           yticklabels=[], zticklabels=[])
    for label, row in dtf_[["x","y","z"]].iterrows():
        x, y, z = row
        ax.text(x, y, z, s=label)

    ## tokenize text
    tokenizer = kprocessing.text.Tokenizer(
        lower=True, split=' ', oov_token="NaN",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
    tokenizer.fit_on_texts(lst_corpus)
    dic_vocabulary = tokenizer.word_index## create sequence
    lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)## padding sequence
    X_train = kprocessing.sequence.pad_sequences(
        lst_text2seq, maxlen=15, padding="post", truncating="post"
        )
    # sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False)
    # plt.show()
    i = 0
    ## list of text: ["I like this", ...]
    len_txt = len(dtf_train["text_clean"].iloc[i].split())
    print("from: ", dtf_train["text_clean"].iloc[i], "| len:", len_txt)
    ## sequence of token ids: [[1, 2, 3], ...]
    len_tokens = len(X_train[i])
    print("to: ", X_train[i], "| len:", len(X_train[i]))
    ## vocabulary: {"I":1, "like":2, "this":3, ...}
    print("check: ", dtf_train["text_clean"].iloc[i].split()[0],
          " -- idx in vocabulary -->",
          dic_vocabulary[dtf_train["text_clean"].iloc[i].split()[0]])
    print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")
    corpus = dtf_test["text_clean"]
    ## create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)
        ## detect common bigrams and trigrams using the fitted detectors
    print("one", lst_corpus[:10])
    lst_corpus = list(bigrams_detector[lst_corpus])
    print("two", lst_corpus[:10])
    lst_corpus = list(trigrams_detector[lst_corpus])
    print("three", lst_corpus[:10])

    # # // to create the bigrams
    # bigram_model = gensim.models.phrases.Phrases(lst_corpus)
    #
    # # // apply the trained model to a sentence
    # bigram_sentences = [u' '.join(bigram_model[unigram_sentence]) for
    #                     unigram_sentence in unigram_sentences]
    #
    # # // get a trigram model out of the bigram
    # trigram_model = gensim.models.phrases.Phrases(bigram_sentences)

    ## text to sequence with the fitted tokenizer
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)
    print(lst_text2seq)
    ## padding sequence
    X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")
    print(lst_corpus[0])
    # print(lst_text2seq)
    print(X_test.shape)
    print(X_test[0])
    ## start the matrix (length of vocabulary x vector size) with all 0s
    embeddings = np.zeros((len(dic_vocabulary)+1, 300))
    for word,idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] =  nlp.wv[word]
        ## if word not in model then skip and the row stays all 0s
        except:
            pass
    word = "data"
    print("dic[word]:", dic_vocabulary[word], "|idx")
    print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape, "|vector")
    ## code attention layer

    ## input
    x_in = layers.Input(shape=(15,))## embedding
    x = layers.Embedding(input_dim=embeddings.shape[0],
                         output_dim=embeddings.shape[1],
                         weights=[embeddings],
                         input_length=15, trainable=False)(x_in)## apply attention
    x = attention_layer(x, neurons=15)## 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2,
                             return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)## final dense layers
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(3, activation='softmax')(x)## compile
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    ## encode y
    dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
    inverse_dic = {v:k for k,v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])
    ## train
    training = model.fit(
        x=X_train, y=y_train, batch_size=256, epochs=10, shuffle=True,
        verbose=0, validation_split=0.3)
    ## plot loss and accuracy
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
         ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    # plt.show()
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
    ## select observation
    i = 0
    txt_instance = dtf_test["text"].iloc[i]## check true value and predicted value
    print("True:", y_test[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))
    ## show explanation

    # explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=3)
    # print(explained)
    # explained.save_to_file("tmp2.html")
    # die
    ### 1. preprocess input
    corpus = []
    print(dtf_test["text"])
    print(txt_instance)
    die
    for i in [re.sub(r'[^\w\s]','', txt_instance.lower().strip())]:
        words = i.split()
        grams = [" ".join(words[i:i+1]) for i in range(0, len(words), 1)]
        corpus.append(grams)
    corpus = list(bigrams_detector[corpus])
    corpus = list(trigrams_detector[corpus])
    X_instance = preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(corpus), maxlen=20, padding="post",
        truncating="post"
        )
    ### 2. get attention weights
    layer = [layer for layer in model.layers if "attention" in layer.name][0]
    func = K.function([model.input], [layer.output])
    weights = func(X_instance)[0]
    weights = np.mean(weights, axis=2).flatten()
    ### 3. rescale weights, remove null vector, map word-weight
    weights = preprocessing.MinMaxScaler(
        feature_range=(0,1)
        ).fit_transform(np.array(weights).reshape(-1,1)).reshape(-1)
    # print(weights)
    # for n,idx in enumerate(X_instance[0]):
    #     if idx != 0:
    #         print(weights[n])
    weights = [weights[n] for n,idx in enumerate(X_instance[0]) if idx != 0]
    # print(tokenizer.word_index.keys())
    print(len(weights), weights)
    print(len(lst_corpus[0]), lst_corpus[0])
    if len(weights) == len(lst_corpus[0]):
        print("############### PASS")
        for n, word in enumerate(lst_corpus[0]):
            if word in tokenizer.word_index.keys():
                print({word:weights[n]})

        dic_word_weigth = {
            word:weights[n] for n,word in enumerate(lst_corpus[0]) if word in tokenizer.word_index.keys()
            }
        ### 4. barplot
        if len(dic_word_weigth) > 0:
           dtf = pd.DataFrame.from_dict(dic_word_weigth, orient='index',columns=["score"])
           dtf.sort_values(by="score", ascending=True).tail().plot(kind="barh", legend=False).grid(axis='x')
           # plt.show()
        else:
           print("--- No word recognized ---")
        ### 5. produce html visualization
        text = []
        for word in lst_corpus[0]:
            weight = dic_word_weigth.get(word)
            if weight is not None:
                 text.append('<b><span style="background-color:rgba(100,149,237,' + str(weight) + ');">' + word + '</span></b>')
            else:
                 text.append(word)
        text = ' '.join(text)
        ### 6. visualize on notebook
        with open("tmp2.html", mode="w") as outfile:
            outfile.write(text)
    # print("\033[1m"+"Text with highlighted words")
    # from IPython.core.display import display, HTML
    # display(HTML(text))

if __name__ == '__main__':
    main()
