#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import twokenize
import sklearn.feature_extraction
from nltk.classify.scikitlearn import SklearnClassifier
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import textPreprocessor01
import nltk
from nltk.stem import *
from nltk.probability import FreqDist
from nltk.corpus import sentiwordnet as swn

from gensim.models import word2vec
# import word2vecReader
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# TODO: load training data
def read_training_data(training_data):
    id_gts = {}
    with open(training_data, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.split('\t')
            tweetid = fields[0]
            gt = fields[1]
            content = fields[2].strip()
            id_gts[tweetid] = gt, content
    return id_gts


# traindic = read_training_data('twitter-training-data1.txt')
traindic = read_training_data('twitter-training-data_small.txt')
# traindic = read_training_data('twitter-training-data.txt')


# input here
def perprocessing(tdic):
    new_dic = {}
    for line in tdic:
        id = line
        gt = tdic[line][0]
        raw = ' '.join(twokenize.tokenizeRawTweetText(tdic[line][1]))
        text = twokenize.normalizeTextForTagger(raw)
        text_tk = twokenize.tokenize(text)
        telist = []
        for word in text_tk:
            word = word.lower()
            ps = nltk.stem.PorterStemmer()
            word = ps.stem(word)
            telist.append(word)
        newtext = ' '.join(telist)
        # print(newtext)
        newtext = textPreprocessor01.replaceall(newtext)
        new_dic[id] = gt, newtext
    return new_dic


# print(new_dic)
def get_train_corpus(new_dic):
    traincorpus = []
    for line in new_dic:
        traincorpus.append(new_dic[line][1])
    return traincorpus


def get_split_corpus(new_dic):
    split_traincorpus = []
    for line in new_dic:
        split_traincorpus.append(new_dic[line][1].split())
    return split_traincorpus


# tdic = read_training_data('twitter-training-data.txt')
# print(tdic)
# for i in tdic:
#     print(i) #id
#     print(tdic[i])
#     print(tdic[i][0])  # gt. positive/negative
#     print(tdic[i][1])  # content
# print(corpus)
# print(split_corpus)


# TODO extract features

def get_vect():
    vect = CountVectorizer(stop_words='english',lowercase=True)
    # vect = CountVectorizer(stop_words='english', min_df=  ,lowercase=True)
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vect.fit_transform(train_corpus)
    return vect, X


def get_train_ngrams():
    # vectorizer = CountVectorizer(stop_words='english')
    # vect = CountVectorizer(stop_words='english')
    # # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    # X = vect.fit_transform(corpus)
    # print(vectorizer.vocabulary_)
    X = get_vect()[1]
    # print(vectorizer.vocabulary_.keys())
    # print('ngram----')
    # print(X.todense())
    # print(len(X.todense()))
    # X.todense()
    # print(X.toarray())
    return np.array(X.todense())


def get_test_ngrams(corpus):
    vect = get_vect()[0]
    X = vect.transform(corpus)
    b = X.todense()
    return np.array(b)


def get_tfidf(corpus):
    # vectorizer = CountVectorizer(stop_words='english')
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    vect = get_vect()[0]
    tfidf = TfidfVectorizer(vocabulary=list(vect.vocabulary_.keys()), min_df=0.6, lowercase=True, stop_words='english')
    tfs = tfidf.fit_transform(corpus)
    # X = vect.fit_transform(corpus)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.vocabulary_.keys())
    tt = tfs.todense()
    # print('tfid..')
    # print(len(tt))
    return np.array(tt)


# maybe it's wrong
def wordembedding(split_corpus):
    # model = word2vec.Word2Vec(sentences, \
    #                           workers=num_workers, \
    #                           size=num_features, \
    #                           min_count=min_word_count, \
    #                           window=context,
    #                           sample=downsampling)
    model = word2vec.Word2Vec(split_corpus, size=50, min_count=1)

    # To make the model memory efficient
    model.init_sims(replace=True)
    # Saving the model for later use. Can be loaded using Word2Vec.load()
    model_name = "wordembedding_features"
    model.save(model_name)
    # print(model['may'])
    # print('word embedding --------------')
    # print(model.wv.syn0)
    # print(model.wv.vocab)
    # print(len(model.wv.vocab))
    # print(model.wv.index2word)
    print(len(model.wv.index2word))
    print(len(model.wv.syn0))


# right here
def word_embedding2(split_corpus):
    # print('word embedding2 --------------------')
    all = []
    for i in split_corpus:
        # print(i)
        model = word2vec.Word2Vec([i], size=300, min_count=1)
        # print(model.vocabulary)
        # print(model.wv.vocab)
        # s = model.wv.syn0
        s = model.wv.vectors
        ans = list(map(sum, zip(*s)))  # sum of them
        all.append(ans)
    return np.array(all)


def senti_bi_lexicon(split_corpus):
    def inputfile(file):
        with open(file, 'r') as my_file:
            words = [every_line.rstrip() for every_line in my_file]
            return words

    def count_p_n(mylist):
        pos_num = 0
        neg_num = 0
        positive = inputfile('positive-words.txt')
        negative = inputfile('negative-words.txt')
        p_dic = FreqDist(positive)
        n_dic = FreqDist(negative)
        for word in mylist:
            pos_num += p_dic[word]
            neg_num += n_dic[word]
        return pos_num, neg_num

    P_N = []
    for line in split_corpus:
        p_num_all = n_num_all = 0
        p_n_num = count_p_n(line)
        p_num_all += p_n_num[0]
        n_num_all += p_n_num[1]
        P_N.append([p_num_all, n_num_all])
    # print('sent..')
    # print(len(P_N))
    return np.array(P_N)


def get_url(split_corpus):
    url = []
    for i in split_corpus:
        num = i.count('URLLINK')
        url.append([num])
    # print(url)
    # print(len(url))
    return np.array(url)


def get_mention(split_corpus):
    men = []
    for i in split_corpus:
        num = i.count('USERMENTION')
        men.append([num])
        # print(url)
        # print(len(url))
    return np.array(men)


def get_face(split_corpus):
    face = []
    for i in split_corpus:
        numi = i.count('HAPPYFACE')
        numj = i.count('SADFACE')
        face.append([numi, numj])
        # print(url)
        # print(len(url))
    return np.array(face)


newdic = perprocessing(traindic)
train_corpus = get_train_corpus(newdic)
split_corpus = get_split_corpus(newdic)
# print(split_corpus)

F1 = get_train_ngrams()
F2 = get_tfidf(train_corpus)
F3 = senti_bi_lexicon(split_corpus)
# print(F3)
F4 = word_embedding2(split_corpus)
# print(F4)
F5 = get_url(split_corpus)
# print(F5)
F6 = get_mention(split_corpus)
F7 = get_face(split_corpus)
# print(F7)
# print(F7)
# X = np.concatenate((F3, F4, F5, F7), axis=1)
# X = np.concatenate((F3, F1, F5, F7), axis=1)
# print(X)
# labels_to_array = {"positive": 1, "negative": -1, "neutral": 0}
labels_to_array = {"positive": 0, "negative": 2, "neutral": 1}
labels = [labels_to_array[newdic[tweet][0]] for tweet in newdic]
# print(labels)
# print('5.Y..')
Y = np.array(labels)
# X3 = F5
# print(F3)
# X = F1
# X = F2
# X = F4
# X5 = F5
# X35 = np.concatenate((X3, X5), axis=1)

# X = F5
# X = F6
# print(F5)
# print(F6)
# X = np.concatenate((F1, F2, F3, F4, F5, F6, F7), axis=1)
# X = np.concatenate((F1, F3), axis=1)
# X = F7
for classifier in ['MNB','Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'KNN']:
# for classifier in ['Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'KNN']:

    # You may rename the names of the classifiers to something more descriptive
    if classifier == 'Naive Bayes':
        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
        # X = F1
        # Y = Y.reshape(Y.size, 1)
        X = np.concatenate((F3, F5, F4, F7), axis=1)
        model = GaussianNB()
        model.fit(X, Y)
        # vec = DictVectorizer(sparse=False)
        # svm_clf = svm.SVC(kernel='linear')
        # model = Pipeline([('vectorizer', vec), ('svm', svm_clf)])
        # model = svm.SVC()

    elif classifier == 'MNB':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3

        # model = SklearnClassifier(MultinomialNB())
        # model.train(X)
        X = F1

        # base_model = MultinomialNB(alpha=1)
        # model = OnevsRestClassifier(base_model).fit(X,Y)
        model = MultinomialNB(alpha=1,  class_prior=None, fit_prior=True)
        # model.fit(np.array(X), np.array(Y))
        # print(X)
        model.fit(X, Y)


        # joblib.dump(model, 'F3_and_SVM.pkl')



    elif classifier == 'Decision Tree':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
        # X = F3
        X = np.concatenate((F3, F4, F7), axis=1)
        model = tree.DecisionTreeClassifier()
        model.fit(X, Y)

        # lr = Pipeline([('sc', StandardScaler()),
        #                ('clf', LogisticRegression())])

        # y_hat = lr.predict(x_test)
        # y_hat = y_hat.reshape(x1.shape)

    elif classifier == 'Logistic Regression':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3
        X = np.concatenate((F3, F4,F5, F7), axis=1)
        model = LogisticRegression()
        # model.fit(x, y.ravel())
        model.fit(X, Y)

    elif classifier == 'Random Forest':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        # forest = RandomForestClassifier(criterion='entropy',
        # n_estimators = 10,
        # random_state = 1,
        # n_jobs = 2)
        X = F2
        model.fit(X, Y)

    elif classifier == 'KNN':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3
        model = KNeighborsClassifier(n_neighbors=5, p=2)
        # model = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        X = F3
        model.fit(X, Y)

    # mymodel = model
    for testset in testsets.testsets:
        # TODO: classify tweets in test set
        # if testset == 'twitter-test1.txt':
        test = read_training_data(testset)

        testdic = perprocessing(test)
        t_corpus = get_train_corpus(testdic)
        ts_corpus = get_split_corpus(testdic)

        tF1 = get_test_ngrams(t_corpus)
        tF2 = get_tfidf(t_corpus)
        tF3 = senti_bi_lexicon(ts_corpus)
        tF4 = word_embedding2(ts_corpus)
        tF5 = get_url(ts_corpus)
        tF6 = get_mention(ts_corpus)
        tF7 = get_face(ts_corpus)

        if classifier == 'Naive Bayes':
            Xt = np.concatenate((tF3, tF4, tF5, tF7), axis=1)
        elif classifier == 'MNB':
            Xt = tF1
        elif classifier == 'Logistic Regression':
            Xt = np.concatenate((tF3, tF4, tF5, tF7), axis=1)
            # Xt = tF4
        elif classifier == 'KNN':
            Xt = tF3
        elif classifier == 'Decision Tree':
            Xt = np.concatenate((tF3, tF7, tF4), axis=1)
        elif classifier == 'Random Forest':
            Xt = tF2
        # ans_num = model.predict(t_F3)
        # model = joblib.load('F3_and_SVM.pkl')
        # ans_num = model.predict(t_F3)
        # ans_num = model.predict(t_F5)
        # Xt = np.concatenate((tF1, tF2, tF3, tF4, tF5, tF6), axis=1)
        # Xt = np.concatenate((tF1, tF2, tF3, tF4, tF5, tF6, tF7), axis=1)
        # Xt = np.concatenate((tF1, tF3, tF5, tF6, tF7), axis=1)
        # Xt = np.concatenate((tF3, tF1, tF5, tF7), axis=1)
        # Xt = np.concatenate((tF1), axis=1)
        # Xt = tF7
        # Xt = tF1
        ans_num = model.predict(Xt)
        # ans_num = model.predict(t_F1)
        # ans_num = model.predict(t_F2)
        # # print(ans)
        # # print(len(ans))
        array_to_labels = {0: "positive", 2: "negative", 1: "neutral"}
        labels = [array_to_labels[i] for i in ans_num]
        # # print(labels)
        # # ans_dic = {}
        predictions = dict(zip(list(testdic.keys()), labels))
        # print(ans_dictionary)

        # predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '
        # 102313285628711403': 'neutral', '653274888624828198': 'neutral'}
        # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        # predictions = ans_dictionary
        evaluation.evaluate(predictions, testset, classifier)
        evaluation.confusion(predictions, testset, classifier)
