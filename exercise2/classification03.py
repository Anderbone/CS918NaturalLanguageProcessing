#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import twokenize
import sklearn.feature_extraction
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
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree

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
# traindic = read_training_data('twitter-training-data_small.txt')
traindic = read_training_data('twitter-training-data.txt')


# input here
# test1 = read_training_data('twitter-test1.txt')
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
            # word = nltk.stem.SnowballStemmer(word)
            telist.append(word)
        # 	return ''.join(ans)
        # newtext = ?telist
        # newtext = ' '.join(text_tk)
        newtext = ' '.join(telist)
        # print(newtext)
        newtext = textPreprocessor01.replaceall(newtext)
        new_dic[id] = gt, newtext
        # print(type(tdic[line][1]))
        # print(line)
        # print(type(line))
        # print(type(newtext))
        # print(newtext)
    return new_dic

# print(new_dic)
def get_train_corpus(new_dic):
    traincorpus = []
    for line in new_dic:
        # print('after perprocessing')
        # print(line)
        # print(new_dic[line])
        # print(new_dic[line][1])
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
    vect = CountVectorizer(stop_words='english')
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vect.fit_transform(train_corpus)
    return vect, X

def get_ngrams(corpus):
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

def get_ngrams2(corpus):
    # dic = {}
    all = []
    for i in corpus:
        X = vect.fit_transform([i])
        d = X.todense()
        fea = np.array(d)
        all.append(fea)
    print(all)
    return all


def get_tfidf(corpus):
    # vectorizer = CountVectorizer(stop_words='english')
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    vect = get_vect()[0]
    tfidf = TfidfVectorizer(vocabulary=list(vect.vocabulary_.keys()),min_df=0.5, lowercase=True, stop_words='english')
    tfs = tfidf.fit_transform(corpus)
    # X = vect.fit_transform(corpus)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.vocabulary_.keys())
    tt = tfs.todense()
    return np.array(tt)

def get_test_tfidf(corpus):
    vect = get_vect()[0]
    tfidf = TfidfVectorizer(vocabulary=list(vect.vocabulary_.keys()), stop_words='english')
    tfs = tfidf.transform(corpus)
    tt = tfs.todense()
    return np.array(tt)


    # feature_names = tfidf.get_feature_names()
    # #  here, newdic.keys is not from corpus. maybe don't need.. or revised. use dic instead of corpus to input
    # corpus_index = [n for n in list(newdic.keys())]
    # rows, cols = tfs.nonzero()
    # for row, col in zip(rows, cols):
    #     print((feature_names[col], corpus_index[row]), tfs[row, col])
    # df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)
    # print(df)

# getgrams_tfidf()

# maybe it's wrong
def wordembedding(split_corpus):
    # model = word2vec.Word2Vec(sentences, \
    #                           workers=num_workers, \
    #                           size=num_features, \
    #                           min_count=min_word_count, \
    #                           window=context,
    #                           sample=downsampling)
    model = word2vec.Word2Vec(split_corpus, min_count=1)

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
    # mod el.accuracy
    # a = model.similarity('woman', 'man')
    # print(a)

# right here
def word_embedding2(split_corpus):
    print('word embedding2 --------------------')
    all = []
    for i in split_corpus:
        model = word2vec.Word2Vec([i], min_count=1)
        # print(model.vocabulary)
        # print(model.wv.vocab)

        s = model.wv.syn0

        # print(s)
        # print(len(i))   # i is each tweet.  len here is 20,25,22,14
        # print(len(s))  # len here is 22 30 34 28. why the length here is not equal to original tweet.
        ans = list(map(sum, zip(*s)))  # sum of them
        # print(ans)
        # print(len(ans))
        all.append(ans)
    # print(len(all))  # 4
    # print(all)
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
        # p_n_dif = p_n_num[0] - p_n_num[1]
        # if p_n_dif > 0:
        #     P_N.append(1)  # positive
        # elif p_n_dif < 0:
        #     P_N.append(-1)  # negative
        # else:
        #     P_N.append(0)  # neutral
        P_N.append([p_num_all, n_num_all])
    # print('senti_binary_lex--------------------')
    # print(P_N)
    # print(len(P_N))
    return np.array(P_N)


newdic = perprocessing(traindic)
train_corpus = get_train_corpus(newdic)
split_corpus = get_split_corpus(newdic)
# print(split_corpus)

# F1 = get_ngrams2(train_corpus)
# F1 = get_ngrams(train_corpus)
# F1 = np.array(f1)
F2 = get_tfidf(train_corpus)
# F2 = np.array(f2)

# F3 = senti_bi_lexicon(split_corpus)
# F3 = np.array(f3)
# F4 = word_embedding2(split_corpus)
# F4 = np.array(f4)

labels_to_array = {"positive": 1, "negative": -1, "neutral": 0}
labels = [labels_to_array[newdic[tweet][0]] for tweet in newdic]
# print(labels)
Y = np.array(labels)
# X = F3
# X = F1
X = F2

for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'myclassifier1':
        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
        # X = F1
        # Y = Y.reshape(Y.size, 1)
        #
        model = GaussianNB()
        #
        # model.fit(X, Y.ravel())
        #
        # # print(clf.predict([[]]))
        #
        # ans_num = clf.predict(t1_F3)
        # # print(ans)
        # # print(len(ans))
        # array_to_labels = {1:"positive" ,  -1 :"negative",  0:"neutral"}
        # labels = [array_to_labels[i] for i in ans_num]
        # # print(labels)
        # # ans_dic = {}
        # ans_dictionary = dict(zip(list(test1dic.keys()), labels))
        # print(ans_dictionary)

        # num_right = 0
        # for count,i in enumerate(newdic):
        #     # print(i)
        #     # print(count)
        #     # print(newdic[i][0])
        #     # print(type(sentlist[count]))
        #     if sentlist[count] == 1 and newdic[i][0] == 'positive':
        #         # print('right ans')
        #         num_right += 1
        #     if sentlist[count] == 0 and newdic[i][0] == 'neutral':
        #         num_right += 1
        #     if sentlist[count] == -1 and newdic[i][0] == 'negative':
        #         num_right += 1
        # print(num_right/len(sentlist))


        # model = GaussianNB()
        # model.fit(X, Y)
        # vec = DictVectorizer(sparse=False)
        # svm_clf = svm.SVC(kernel='linear')
        # model = Pipeline([('vectorizer', vec), ('svm', svm_clf)])
        # model = svm.SVC()
        model.fit(X, Y)
        # joblib.dump(model, 'F3_and_SVM.pkl')



    elif classifier == 'myclassifier2':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
        # X = F3
        model = tree.DecisionTreeClassifier()
        model.fit(X, Y)

        # lr = Pipeline([('sc', StandardScaler()),
        #                ('clf', LogisticRegression())])

        # y_hat = lr.predict(x_test)  # 预测值
        # y_hat = y_hat.reshape(x1.shape)



        # X = F1
        # # model = GaussianNB()
        # # model.fit(X, Y)
        # vec = DictVectorizer(sparse=False)
        # svm_clf = svm.SVC(kernel='linear')
        # model = Pipeline([('vectorizer', vec), ('svm', svm_clf)])
        # model.fit(X, Y)
        # joblib.dump(vec_clf, 'vectorizer_and_SVM.pkl')
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3
        # x = F3
        # y = Y
        model = LogisticRegression()
        # model.fit(x, y.ravel())
        model.fit(X, Y)


    for testset in testsets.testsets:
        # TODO: classify tweets in test set
        test = read_training_data(testset)

        testdic = perprocessing(test)
        t_corpus = get_train_corpus(testdic)
        ts_corpus = get_split_corpus(testdic)

        # t_F1 = get_test_ngrams(t_corpus)
        # t_F1 = np.array(t_f1)
        # t_F2 = get_test_tfidf(t_corpus)
        t_F2 = get_tfidf(t_corpus)

        # t_F3 = senti_bi_lexicon(ts_corpus)
        # t_F3 = np.array(t_f3)

        # t_F4 = word_embedding2(ts_corpus)
        # t_F4 = np.array(t_f4)

        # ans_num = model.predict(t_F3)
        # model = joblib.load('F3_and_SVM.pkl')
        # ans_num = model.predict(t_F3)
        # ans_num = model.predict(t_F1)
        ans_num = model.predict(t_F2)
        # # print(ans)
        # # print(len(ans))
        array_to_labels = {1:"positive",  -1:"negative",  0:"neutral"}
        labels = [array_to_labels[i] for i in ans_num]
        # # print(labels)
        # # ans_dic = {}
        predictions = dict(zip(list(testdic.keys()), labels))
        # print(ans_dictionary)


        # predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        # predictions = ans_dictionary
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
