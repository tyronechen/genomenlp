#!/usr/bin/python
# take custom formatted bedfile, break into kmers for classification with lime
import argparse
from math import floor
import os
import re
from textwrap import wrap
import warnings
from warnings import warn
# from joblib import parallel_backend, Parallel, delayed, dump, load
import json
# import gensim
# from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt
import numpy as np
# from lime.lime_text import LimeTextExplainer
import pandas as pd
# import screed
# from screed import ScreedDB
# import seaborn as sns
# import sklearn.ensemble
from sklearn import metrics, manifold, model_selection, preprocessing
# from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# import sklearn.metrics
from utils import bootstrap_seq, build_kmers, plot_roc, map_synthetic_real, \
    select_features, show_embedding_word, show_sample_vector, show_acc_loss, \
    show_summary_stats, _map_synthetic_real, _tokenise_seqs, _write_output
# from tensorflow.keras import models, layers, backend, \
#     preprocessing as k_preprocessing
from dna2vec.multi_k_model import MultiKModel

def main():
    parser = argparse.ArgumentParser(
        description='Take custom formatted bedfile, classify kmers with lime.'
    )
    parser.add_argument('-i', '--infile_path', type=str, default=None,
                        help='path to embeddings file with data')
    parser.add_argument('-b', '--block_size', type=int, default=2,
                        help='size of block to permute null seqs (DEFAULT: 2)')
    parser.add_argument('-c', '--control_set', type=str, default=None,
                        help='use an existing bed-like file as a null set')
    parser.add_argument('-g', '--genome', type=str, default=None,
                        help='input is provided as a fasta file')
    parser.add_argument('--train_data', type=str, default=None,
                        help='use an existing bed-like file as a training set')
    parser.add_argument('--test_data', type=str, default=None,
                        help='use an existing bed-like file as a testing set')
    parser.add_argument('-s', '--sample_classes', type=str, nargs="+",
                        default=["control", "test"],
                        help='sample classes (DEFAULT: ["control", "test"])')
    parser.add_argument('-w', '--window_size', type=int, default=5,
                        help='size of sliding window to take (DEFAULT: 5)')
    parser.add_argument('-e', '--epoch', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('-m', '--model_load', type=str, default=None,
                        help='load model from this path (DEFAULT: None)')
    parser.add_argument('-v', '--vectorizer_load', type=str, default=None,
                        help='load vectorizer from this path (DEFAULT: None)')
    parser.add_argument('-n', '--ngram_count', type=int, default=1,
                        help='number of kmers to include in ngram (DEFAULT: 1)')
    parser.add_argument('-fs', '--feature_selection', type=float, default=None,
                        help='chi square test for feature:target independence (DEFAULT: None)')
    parser.add_argument('-t', '--threads', type=int, default=2,
                        help='number of cpus to use (DEFAULT: 2)')
    parser.add_argument('-f', '--force_if_lowscore', action="store_true",
                        help='force continue even if acc low (DEFAULT: False)')
    parser.add_argument('-d', '--display_count', type=int, default=0,
                        help='number of test data to show (DEFAULT: 0 [ALL])')
    parser.add_argument('-o', '--outfile_dir', type=str, default=None,
                        help='write html output to this dir (DEFAULT: None)')
    parser.add_argument('-p', '--hide_progressbar', action="store_true",
                        help='hide the progress bar (DEFAULT: False)')

    args = parser.parse_args()
    infile_path = args.infile_path
    threads = args.threads
    outdir = args.outfile_dir
    ksize = args.window_size
    load_model = args.model_load
    load_vectorizer = args.vectorizer_load
    train_data = args.train_data
    test_data = args.test_data
    ngram = args.ngram_count
    control = args.control_set
    sample_classes = args.sample_classes
    display = args.display_count
    hide = args.hide_progressbar
    epoch = args.epoch
    feature_selection = args.feature_selection
    genome = args.genome

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if load_model and train_data or \
        load_model and infile_path or \
        train_data and infile_path:
        message = " ".join([
            "model_load, train_data, infile_path are mutually exclusive!"
            "Priority is given to model_load > train_data > infile_path,"
            "all arguments lower in the sequence will be overriden!"
            ])
        warn(message)

    # this bed-like file is a custom format with the header removed from:
    # http://regulondb.ccg.unam.mx/menu/download/datasets/files/PromoterSet.txt
    # if train_data:
    #     data = pd.read_csv(train_data, sep="\t", header=None)
    # elif infile_path:
    #     data = pd.read_csv(infile_path, sep="\t", header=None)

    mk_model = MultiKModel(infile_path)
    print(mk_model.aggregate.vector_size)
    print(mk_model.aggregate.vectors)
    print(mk_model.aggregate.get_normed_vectors())

    # model = gensim.models.word2vec.Word2Vec(
    #     sentences=unigrams_train,
    #     vector_size=300,
    #     window=20,
    #     min_count=1,
    #     sg=1, # 1 for skip-gram, 0 for cbow
    #     epochs=30,
    #     workers=threads
    #     )
    # # 3d embedding of a word of interest
    # word = unigrams_train[0][0]
    word = "AAAAA"
    words = [
        'GTTTTT', 'GGCGTT', 'ATACA', 'TTTCAG', 'CTGAAA', 'CGCTGC', 'CAGCGG', 'CACAC', 'GTGTG', 'CCAGCC', 'CGGCAG', 'TCATCA', 'TCCCC', 'GGCTA', 'AGCGCC', 'AACTGG', 'CCCGA', 'TTTTCA', 'GGGGA', 'GACTT', 'TGCCGG', 'CTACC', 'CCGGCA', 'GCAGCG', 'GCTCC', 'GGCGCA', 'CTGCCG', 'CTCAG', 'CCCCG', 'GGCTGG', 'GGAGC', 'AGGTC', 'GGCGCT', 'AGCCAG', 'ACGCCA', 'CTGATG', 'CGAGA', 'TTCGCC', 'GGTAG', 'TCTCG', 'ACGTA', 'TACGT', 'CGCGCC', 'CGGGG', 'TCCTT', 'CATCAG', 'CCAGTT', 'CTGGCT', 'TGGGA', 'CTTAC', 'AGCTC', 'CCGCCG', 'GACCT', 'GCGTTT', 'TACAA', 'AAGGA', 'CTTTTT', 'GCCTGC', 'TTGTA', 'AATAG', 'GGCGCG', 'TCCGGC', 'CTGAG', 'TGGAC', 'TGGCAA', 'GGCGAA',
        ]
    words = 'GTTTTT'
    x = mk_model.data[6].model

    from sklearn.decomposition import PCA

    def show_embedding_word(model, word: str, savefig: str="./3d_embedding.pdf"):
        fig = plt.figure()
        tot_words = [word] + [x[0] for x in model.most_similar(word, topn=20)]
        X = model[tot_words]
        pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
        X = pca.fit_transform(X)
        dtf_ = pd.DataFrame(X, index=tot_words, columns=["x","y","z"])
        dtf_["input"] = 0
        dtf_["input"].iloc[0:1] = 1
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
        plt.title(word)
        plt.show()

    show_embedding_word(x, words, savefig="./3d_embedding.pdf")

    def display_pca_scatterplot(model, words=None, sample=0):
        if words == None:
            if sample > 0:
                words = np.random.choice(model.index_to_key, sample)
            else:
                words = [ word for word in mk_model.index_to_key ]
        print(words)
        word_vectors = np.array([model.vector(w) for w in words])

        twodim = PCA().fit_transform(word_vectors)[:,:2]

        plt.figure(figsize=(6,6))
        plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
        for word, (x,y) in zip(words, twodim):
            plt.text(x+0.05, y+0.05, word)
        plt.show()

    # display_pca_scatterplot(x, words=words, sample=0)

    # show_embedding_word(
    #     model=mk_model.model, word=word, savefig="/".join([outdir, "3d_embed.pdf"])
    #     )
    #
    # # print sample vectors for debugging
    # show_sample_vector(train, X_train, vocab)
    #
    # ## start the matrix (length of vocabulary x vector size) with all 0s
    # embeddings = np.zeros((len(vocab)+1, 300))
    # for word, idx in vocab.items():
    #     ## update the row with vector
    #     try:
    #         embeddings[idx] = model[word]
    #     ## if word not in model then skip and the row stays all 0s
    #     except:
    #         pass
    #
    # def _attention_layer(inputs, neurons):
    #     x = layers.Permute((2,1))(inputs)
    #     x = layers.Dense(neurons, activation="softmax")(x)
    #     x = layers.Permute((2,1), name="attention")(x)
    #     x = layers.multiply([inputs, x])
    #     return x
    #
    # ## input
    # x_in = layers.Input(shape=(20,))## embedding
    # x = layers.Embedding(input_dim=embeddings.shape[0],
    #                      output_dim=embeddings.shape[1],
    #                      weights=[embeddings],
    #                      input_length=20, trainable=False)(x_in)
    # ## apply attention
    # x = _attention_layer(x, neurons=20)
    # ## 2 layers of bidirectional lstm
    # x = layers.Bidirectional(layers.LSTM(units=20, dropout=0.2,
    #                          return_sequences=True))(x)
    # x = layers.Bidirectional(layers.LSTM(units=20, dropout=0.2))(x)
    # x = layers.Dense(64, activation='relu')(x)
    # y_out = layers.Dense(2, activation='softmax')(x)## compile
    # model = models.Model(x_in, y_out)
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])
    # model.summary()
    #
    # ## encode y
    # y_train = train["target"].values
    # y_test = test["target"].values
    # y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
    # inverse_map = {v:k for k,v in y_mapping.items()}
    # y_train = np.array([inverse_map[y] for y in y_train])
    #
    # trained = model.fit(
    #     x=X_train, y=y_train, batch_size=256, epochs=10, shuffle=False,
    #     verbose=0, validation_split=0.1
    #     )
    # print(trained.history)
    # plt.plot(trained.history['accuracy'])
    # plt.plot(trained.history['loss'])
    # plt.plot(trained.history['val_accuracy'])
    # plt.plot(trained.history['val_loss'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['accuracy', 'loss', 'val_accuracy', 'val_loss'], loc='upper left')
    # plt.savefig("/".join([outdir, "allmetrics.pdf"]), dpi=300)
    #
    # show_acc_loss(trained=trained, savefig="/".join([outdir, "acc_loss.pdf"]))
    # # print(y_train)
    # # y_train = sklearn.preprocessing.LabelBinarizer().fit_transform(y_train)
    # # print(y_train)
    # predicted_prob = model.predict(X_test)
    # predicted = [y_mapping[np.argmax(pred)] for pred in predicted_prob]
    #
    # show_summary_stats(
    #     y_test=y_test, predicted=predicted, predicted_prob=predicted_prob,
    #     savefig="/".join([outdir, "summary.pdf"])
    #     )
    #
    # ## select observation
    # i = 0
    # txt_instance = " ".join(test["data"].iloc[i])
    # print(txt_instance)
    # ## check true value and predicted value
    # print("Real:", y_test[i],
    #       "\nPred:", predicted[i],
    #       "\nProb:", round(np.max(predicted_prob[i]),2))
    #
    # ## show explanation
    # ### 1. preprocess input
    # corpus = []
    # for string in [re.sub(r'[^\w\s]','', txt_instance.lower().strip())]:
    #     words = string.split()
    #     grams = [" ".join(words[i:i+1]) for i in range(0, len(words), 1)]
    #     corpus.append(grams)
    #
    # if ngram in [2, 3]:
    #     corpus = list(bigrams_detector[corpus])
    #     if ngram is 3:
    #         corpus = list(trigrams_detector[corpus])
    #
    # X_instance = k_preprocessing.sequence.pad_sequences(
    #     tokenizer.texts_to_sequences(corpus), maxlen=20, padding="post",
    #     truncating="post"
    #     )
    #
    # ### 2. get attention weights
    # layer = [layer for layer in model.layers if "attention" in layer.name][0]
    # func = backend.function([model.input], [layer.output])
    # weights = func(X_instance)[0]
    # weights = np.mean(weights, axis=2).flatten()
    #
    # ### 3. rescale weights, remove null vector, map word-weight
    # weights = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(
    #     np.array(weights).reshape(-1,1)
    #     ).reshape(-1)
    # print(weights)
    # weights = [weights[n] for n,idx in enumerate(X_instance[0]) if idx != 0]
    # print(weights)
    # weights = {word:weights[n] for n,word in enumerate(corpus[0]) if word in
    #            tokenizer.word_index.keys()}
    # print(weights)
    #
    # ### 4. barplot
    # if len(weights) > 0:
    #     dtf = pd.DataFrame.from_dict(weights, orient='index', columns=["score"])
    #     dtf.sort_values(by="score", ascending=True, inplace=True)
    #     print(dtf.head())
    #     print(dtf.tail())
    #     dtf.plot(kind="barh", legend=False).grid(axis='x')
    #     # plt.show()
    #
    # ### 5. produce html visualization
    # text = []
    # for word in corpus[0]:
    #     weight = weights.get(word)
    #     if weight is not None:
    #          text.append(
    #              '<b><span style="background-color:rgba(100,149,237,' + str(weight) + ');">' + word + '</span></b>'
    #              )
    #     else:
    #          text.append(word)
    # text = ' '.join(text)
    #
    # ### 6. visualize on notebook
    # with open("foo.html", mode="w") as outfile:
    #     outfile.write(text)
    #
    # import sys
    # sys.exit()

if __name__ == '__main__':
    main()
