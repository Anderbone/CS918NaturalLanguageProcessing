import re
import json
import nltk
import math
import time
import collections
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import bigrams, trigrams
""" Author: Jiyu Yan
    Student ID: 1851015
    For 918 exercise1
"""

def number_v(mylist):
    return len((set(mylist)))


def lemma(str):
    data_list = []
    lemmatizer = WordNetLemmatizer()
    after_tag = nltk.pos_tag(str.split())
    for word, tag in after_tag:
        if tag.startswith('N'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
        elif tag.startswith('VBZ'):   # don't change 'is' to 'be'
            after_lemma = word
        elif tag.startswith('VBP'):
            after_lemma = word
        elif tag.startswith('V'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)
        elif tag.startswith('J'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.ADJ)
        elif tag.startswith('R'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.ADV)
        else:
            after_lemma = word
        data_list.append(after_lemma)
    return data_list


def freqdic(mylist):
    mydict = collections.defaultdict(lambda: 0)
    for key in mylist:
        mydict[key] += 1
    return mydict


def most_common(mydict):
    return sorted(mydict.items(), key=lambda d: d[1], reverse=True)


def inputfile(file):
    with open(file, 'r') as my_file:
        words = [every_line.rstrip() for every_line in my_file]
        return words


def count_p_n(mylist):
    pos_num = 0
    neg_num = 0
    positive = inputfile('positive-words.txt')
    negative = inputfile('negative-words.txt')
    p_dic = freqdic(positive)
    n_dic = freqdic(negative)
    for word in mylist:
        pos_num += p_dic[word]
        neg_num += n_dic[word]
    return pos_num, neg_num


def all_trigram_withcount(mylist):
    trigram_withcount = most_common(freqdic(list(trigrams(mylist))))
    return trigram_withcount


def all_bigram_withcount(mylist):
    bigram_withcount = most_common(freqdic(list(bigrams(mylist))))
    return bigram_withcount


def all_trigram(trigram_withcount):  # return all trigram sorted as frequency
    trigram_list = []
    for i in trigram_withcount:
        trigram_list.append(i[0])
    return trigram_list


def gen_sentence(tri_list, a, b, leng):
    sen = [a, b]
    sen_go = True

    def gen_sentence_in(tri_list, a, b, leng):
        nonlocal sen_go
        nonlocal sen
        if sen_go is False:
            return sen
        for i in tri_list:
            if i[0] == a and i[1] == b and sen_go:
                sen.append(i[2])
                if len(sen) == leng:
                    sen_go = False
                gen_sentence_in(tri_list, b, i[2], leng)
    gen_sentence_in(tri_list, a, b, leng)
    return sen


def perplex(test_sent, mylist):
    tri_sent = list(trigrams(test_sent))
    ftri = freqdic(list(trigrams(mylist)))
    fbi = freqdic(list(bigrams(mylist)))
    fui = freqdic(mylist)
    log_tri_count = 0
    log_bi_count = 0
    v = number_v(mylist)
    a1 = 0.33
    a2 = 0.33
    a3 = 0.33
    a4 = 0.01
    for tri in tri_sent:
        c3 = ftri[tri]
        c2 = fbi[tri[1:]]
        c1 = fui[tri[2]]
        if c3 != 0:
            tri_temp = c3*a1*c1*v*v + v*v*c2*c2*a2 + a3*c1*c1*c2*v + a4*c1*c2*v
            bi_temp = c1*c2*v*v
        elif c3 == 0 and c2 != 0:
            tri_temp = a2*c2*v*v + a3*c1*c1*v + a4*c1*v
            bi_temp = c1*v*v
        elif c3 == 0 and c2 == 0 and c1 != 0:
            tri_temp = a3*c1 + a4
            bi_temp = v
        elif c1 == 0:
            tri_temp = a4
            bi_temp = v
        log_tri_temp = math.log(tri_temp, 10)
        log_tri_count += log_tri_temp
        log_bi_temp = math.log(bi_temp, 10)
        log_bi_count += log_bi_temp
    p = log_tri_count - log_bi_count
    log_pp = (-1*p) /(len(test_sent))
    pp = pow(10, log_pp)
    return pp


if __name__ == '__main__':
    start = time.time()
    # data_list_all = []
    data_16000 = []
    data_3000 = []
    p_num_all, n_num_all = 0, 0
    pos_sto, neg_sto = 0, 0
    count_line = 0
    pattern_no_url = re.compile(r'(https://|http://|www\.)+(\w|\.|/|\?|=|&|%)*\b')
    pattern_no_num = re.compile(r'[a-z0-9]*[a-z][a-z0-9]*')
    pattern_no_1letter = re.compile(r'[a-z0-9]{2,}')
    with open('signal-news1.jsonl', 'r') as f:
    # with open('testline.jsonl', 'r') as f:
        for line in f.readlines():
            content = json.loads(line).get('content').lower()
            content_no_url = pattern_no_url.sub('', content)
            content_no_num = ' '.join(pattern_no_num.findall(content_no_url))
            content_no_1 = ' '.join(pattern_no_1letter.findall(content_no_num))
            content_alldone_list = lemma(content_no_1)
            # data_list_all.extend(content_alldone_list)
            if count_line < 16000:
            # if count_line < 1:
                data_16000.extend(content_alldone_list)
            else:
                data_3000.extend(content_alldone_list)
            count_line += 1
            p_n_num = count_p_n(content_alldone_list)
            p_num_all += p_n_num[0]
            n_num_all += p_n_num[1]
            p_n_dif = p_n_num[0] - p_n_num[1]
            if p_n_dif > 0:
                pos_sto += 1
            elif p_n_dif < 0:
                neg_sto += 1
    data_list_all = data_16000 + data_3000


    # print(data_list_all)
    print('N:%d' % len(data_list_all))
    print('V:%d' % number_v(data_list_all))
    tri_withcount_all = most_common(freqdic(list(trigrams(data_list_all))))[:25]
    tri_list_all = all_trigram(tri_withcount_all)
    print(tri_list_all)

    print('number of positive words in corpus: %d, negative words: %d' % (p_num_all, n_num_all))
    print('number of positive stories: %d, negative stories: %d' % (pos_sto, neg_sto))

    tri_withcount_16000 = all_trigram_withcount(data_16000)
    tri_list_16000 = all_trigram(tri_withcount_16000)
    sen = gen_sentence(tri_list_16000, 'is', 'this', 10)
    print(sen)

    pp = perplex(data_3000, data_16000)
    print('perplexity of remaining rows is ' + str(pp))
    end = time.time()
    print('cost time'+str(end - start))