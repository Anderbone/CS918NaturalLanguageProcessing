#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import twokenize
import collections
import os
# import sklearn.feature_extraction
# from nltk.classify.scikitlearn import SklearnClassifier
# import sklearn
# import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import textPreprocessor02
import nltk
# from nltk.stem import *
from nltk.probability import FreqDist
# from nltk.corpus import sentiwordnet as swn

# from gensim.models import word2vec
import pickle
import word2vecReader
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn import svm
# from sklearn.externals import joblib
# from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn import preprocessing
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# pickle.
# from nltk.corpus import wordnet
# from nltk.stem.wordnet import WordNetLemmatizer

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
        newtext = textPreprocessor02.replaceall(newtext)
        new_dic[id] = gt, newtext
        # print(newtext)
    return new_dic


# print(new_dic)
def get_train_corpus(new_dic):

    traincorpus = []
    for line in new_dic:
        traincorpus.append(new_dic[line][1])
    # print(traincorpus)
    return traincorpus


def get_split_corpus(new_dic):
    split_traincorpus = []
    for line in new_dic:
        split_traincorpus.append(new_dic[line][1].split())
    return split_traincorpus

# TODO extract features

def get_vect():
    if os.path.isfile('X.pkl') and os.path.isfile('vector.pkl'):
        return
    else:
        vect = CountVectorizer(stop_words='english',ngram_range=(1,2), max_features=17000, lowercase=True)
        # vect = CountVectorizer(stop_words='english', min_df=  ,lowercase=True)
        # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        X = vect.fit_transform(train_corpus)
        # print(type(X))
        # print(X)
        # print(vect)
        pickle.dump(X,open('X.pkl', 'wb'))
        pickle.dump(vect, open('vector.pkl', 'wb'))
    # return vect, X

def get_train_ngrams():
    get_vect()
    with open('X.pkl', 'rb') as f:
        X_train = pickle.load(f)
        # return f1
    # X = pickle.load(open('X.pkl','rb'))
        ans = np.array(X_train.todense())
        # pickle.dump(ans, open('F1.pkl', 'wb'))
        return ans
        # return np.array(X.todense())

def get_test_ngrams(corpus):
    # vect = get_vect()[0]
    get_vect()
    with open('vector.pkl', 'rb') as f:
        # f1 = pickle.load(f)
        # return f1
        vect = pickle.load(f)
        X = vect.transform(corpus)
        b = X.todense()
        return np.array(b)

def get_senti_score(split_corpus):
    def senti_file():
        senti_dic = collections.defaultdict(lambda: 0)
        with open('SentiWords_1.1.txt', 'r') as f:
            for line in f:
                fields = line.strip().split('\t')
                senti_dic[fields[0]] = float(fields[1])
            return senti_dic
    mydic = senti_file()
    all_socre = []
    for line in split_corpus:
        after_tag = nltk.pos_tag(line)
        score_feature = 0
        for word,tag in after_tag:
            if tag.startswith('N'):
                after = word+str('#n')
                score = mydic[after]
            elif tag.startswith('V'):
                after = word+str('#v')
                score = mydic[after]
            elif tag.startswith('J'):
                after = word + str('#a')
                score = mydic[after]
            elif tag.startswith('R'):
                after = word + str('#r')
                score = mydic[after]
            else:
                after = word + str('#n')
                score = mydic[after]
            score_feature += score
        all_socre.append([score_feature])
    ans = np.array(all_socre)
    return ans

def get_tfidf(corpus):
    get_vect()
    with open('vector.pkl', 'rb') as f:
        vect = pickle.load(f)
        tfidf = TfidfVectorizer(vocabulary=list(vect.vocabulary_.keys()), lowercase=True, stop_words='english')
        tfs = tfidf.fit_transform(corpus)
        tt = tfs.todense()
        return np.array(tt)


model_path = "word2vec_twitter_model.bin"
model = word2vecReader.Word2Vec.load_word2vec_format(model_path, binary=True)
def word_embedding(split_corpus, model=model, size=400):
    # using external twitter specific per-trained model
    ans = []
    for line in split_corpus:
        vec = np.zeros(size)
        count = 0.
        for word in line:
            try:
                vec += model[word]
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        ans.append(vec.tolist())
    arr = np.array(ans)
    return arr

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

def get_F2():
    if os.path.isfile('F2.pkl'):
        with open('F2.pkl', 'rb') as f:
            f2 = pickle.load(f)
            return f2
    else:
        ans = get_tfidf(train_corpus)
        pickle.dump(ans, open('F2.pkl', 'wb'))
        return ans
def get_F3():
    if os.path.isfile('F3.pkl'):
        with open('F3.pkl', 'rb') as f:
            f3 = pickle.load(f)
            return f3
    else:
        ans = senti_bi_lexicon(split_corpus)
        pickle.dump(ans, open('F3.pkl', 'wb'))
        return ans
def get_F4():
    if os.path.isfile('F4.pkl'):
        with open('F4.pkl', 'rb') as f:
            f4 = pickle.load(f)
            return f4
    else:
        ans = word_embedding(split_corpus)
        pickle.dump(ans, open('F4.pkl', 'wb'))
        return ans
def get_F5():
    if os.path.isfile('F5.pkl'):
        with open('F5.pkl', 'rb') as f:
            f = pickle.load(f)
            return f
    else:
        ans = get_url(split_corpus)
        pickle.dump(ans, open('F5.pkl', 'wb'))
        return ans
def get_F6():
    if os.path.isfile('F6.pkl'):
        with open('F6.pkl', 'rb') as f:
            f = pickle.load(f)
            return f
    else:
        ans = get_mention(split_corpus)
        pickle.dump(ans, open('F6.pkl', 'wb'))
        return ans
def get_F7():
    if os.path.isfile('F7.pkl'):
        with open('F7.pkl', 'rb') as f:
            f = pickle.load(f)
            return f
    else:
        ans = get_face(split_corpus)
        pickle.dump(ans, open('F7.pkl', 'wb'))
        return ans
def get_F8():
    if os.path.isfile('F8.pkl'):
        with open('F8.pkl', 'rb') as f:
            f = pickle.load(f)
            return f
    else:
        ans = get_senti_score(split_corpus)
        pickle.dump(ans, open('F8.pkl', 'wb'))
        return ans

traindic = read_training_data('twitter-training-data.txt')
newdic = perprocessing(traindic)
train_corpus = get_train_corpus(newdic)
split_corpus = get_split_corpus(newdic)
labels_to_array = {"positive": 0, "negative": 2, "neutral": 1}
labels = [labels_to_array[newdic[tweet][0]] for tweet in newdic]
Y = np.array(labels)

for classifier in ['Logistic Regression0','Logistic Regression1','Logistic Regression2']:

    if classifier == 'Logistic Regression0':
        print('Training ' + classifier)
        if os.path.isfile('LR0.pkl'):
            with open('LR0.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            # F1 = get_train_ngrams()
            F2 = get_tfidf(train_corpus)
            F3 = get_F3()
            # print(F3)
            # F4 = get_F4()
            X = np.concatenate((F2, F3), axis=1)
            # pickle.dump(X, open('X0.pkl', 'wb'))
            model = LogisticRegression(class_weight='balanced')
            model.fit(X, Y)
            pickle.dump(model, open('LR0.pkl', 'wb'))

    elif classifier == 'Logistic Regression1':
        print('Training ' + classifier)
        if os.path.isfile('LR1.pkl'):
            with open('LR1.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            F2 = get_tfidf(train_corpus)
            F3 = get_F3()
            F4 = get_F4()
            X = np.concatenate((F2, F3, F4), axis=1)
            model = LogisticRegression(class_weight='balanced')
            model.fit(X, Y)
            pickle.dump(model, open('LR1.pkl', 'wb'))

    elif classifier == 'Logistic Regression2':
        print('Training ' + classifier)
        if os.path.isfile('LR2.pkl'):
            with open('LR2.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            # F1 = get_train_ngrams()
            F2 = get_tfidf(train_corpus)
            F3 = get_F3()
            F4 = get_F4()
            F5 = get_F5()
            F7 = get_F7()
            F8 = get_F8()
            X = np.concatenate((F2, F3, F4, F5, F7, F8), axis=1)
            model = LogisticRegression(class_weight='balanced')
            model.fit(X, Y)
            pickle.dump(model, open('LR2.pkl', 'wb'))

    for testset in testsets.testsets:
        # TODO: classify tweets in test set
        test = read_training_data(testset)
        testdic = perprocessing(test)
        t_corpus = get_train_corpus(testdic)
        ts_corpus = get_split_corpus(testdic)
        if classifier == 'Logistic Regression0':
            # tF1 = get_test_ngrams(t_corpus)
            tF2 = get_tfidf(t_corpus)
            tF3 = senti_bi_lexicon(ts_corpus)
            # tF4 = word_embedding(ts_corpus)
            Xt = np.concatenate((tF2,tF3), axis=1)
        elif classifier == 'Logistic Regression1':
            tF2 = get_tfidf(t_corpus)
            tF3 = senti_bi_lexicon(ts_corpus)
            tF4 = word_embedding(ts_corpus)
            Xt = np.concatenate((tF2, tF3, tF4), axis=1)
        elif classifier == 'Logistic Regression2':
            # tF1 = get_test_ngrams(t_corpus)
            tF2 = get_tfidf(t_corpus)
            tF3 = senti_bi_lexicon(ts_corpus)
            tF4 = word_embedding(ts_corpus)
            tF5 = get_url(ts_corpus)
            tF7 = get_face(ts_corpus)
            tF8 = get_senti_score(ts_corpus)
            Xt = np.concatenate((tF2, tF3, tF4,tF5, tF7, tF8), axis=1)

        # prob = model.predict_proba(Xt)
        # print(prob)
        ans_num = model.predict(Xt)
        # print(ans_num)
        ans_num = []
        # for each in prob:
        #     pos = each[0]
        #     neu = each[1]+0.05
        #     neg = each[2]+0.05
        #     if pos > neu and pos > neg:
        #         ans = 0
        #     elif  neu > neg:
        #         ans = 1
        #     else:
        #         ans = 2
        #     ans_num.append(ans)

        array_to_labels = {0: "positive", 1: "neutral", 2: "negative"}
        labels = [array_to_labels[i] for i in ans_num]
        predictions = dict(zip(list(testdic.keys()), labels))
        evaluation.evaluate(predictions, testset, classifier)
        evaluation.confusion(predictions, testset, classifier)
