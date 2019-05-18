import re
import json
import nltk
import math
import time
import collections
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import bigrams, trigrams

# no more back off, linear is best!!!!!!!!!!!
def number_v(mylist):
    return len((set(mylist)))


def lemma(str):
    data_list = []
    lemmatizer = WordNetLemmatizer()
    after_tag = nltk.pos_tag(str.split())
    for word, tag in after_tag:
        if tag.startswith('N'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
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


def all_trigram(trigram_withcount):  # return all trigram sorted as frequency and no duplicate
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
    vocabulary = number_v(mylist)
    for tri in tri_sent:
        if ftri[tri] != 0:
            tri_temp = ftri[tri] * 0.37
            bi_temp = fbi[tri[:2]]
        elif ftri[tri] == 0 and fbi[tri[:2]] != 0:
            tri_temp = fbi[tri[:2]] * 0.37
            bi_temp = fui[tri[0]]
        elif ftri[tri] == 0 and fbi[tri[:2]] == 0 and fui[tri[0]] != 0:
            tri_temp = fui[tri[0]] * 0.24
            bi_temp = vocabulary
        elif fui[tri[0]] == 0:
            tri_temp = 1 * 0.02
            bi_temp = vocabulary
        log_tri_temp = math.log(tri_temp, 10)
        log_tri_count += log_tri_temp
        log_bi_temp = math.log(bi_temp, 10)
        log_bi_count += log_bi_temp
    p = log_tri_count - log_bi_count
    print('old perplexity ')
    print(p)
    log_pp = (-1*p) /(len(test_sent))
    pp = pow(10, log_pp)
    return pp

def perplex2(test_sent, mylist):
    tri_sent = list(trigrams(test_sent))
    ftri = freqdic(list(trigrams(mylist)))
    fbi = freqdic(list(bigrams(mylist)))
    fui = freqdic(mylist)
    log_tri_count0 = 0
    log_bi_count0 = 0
    log_tri_count1 = 0
    log_bi_count1 = 0
    log_tri_count2 = 0
    log_bi_count2 = 0
    log_tri_count3 = 0
    log_bi_count3 = 0
    vocabulary = number_v(mylist)
    for tri in tri_sent:
        if ftri[tri] != 0:
            tri_temp0 = ftri[tri]*0.37
            bi_temp0 = fbi[tri[:2]]
            log_tri_temp0 = math.log(tri_temp0, 10)
            log_bi_temp0 = math.log(bi_temp0, 10)
            log_tri_count0 += log_tri_temp0
            log_bi_count0 += log_bi_temp0
        if fbi[tri[:2]] != 0:
            tri_temp1 = fbi[tri[:2]]*0.37
            bi_temp1 = fui[tri[0]]
            log_tri_temp1 = math.log(tri_temp1, 10)
            log_bi_temp1 = math.log(bi_temp1, 10)
            log_tri_count1 += log_tri_temp1
            log_bi_count1 += log_bi_temp1
        if fui[tri[0]] != 0:
            tri_temp2 = fui[tri[0]]*0.24
            bi_temp2 = vocabulary
            log_tri_temp2 = math.log(tri_temp2, 10)
            log_bi_temp2 = math.log(bi_temp2, 10)
            log_tri_count2 += log_tri_temp2
            log_bi_count2 += log_bi_temp2
        if fui[tri[0]] == 0:
            tri_temp3 = 1*0.02
            bi_temp3 = vocabulary
            log_tri_temp3 = math.log(tri_temp3, 10)
            log_bi_temp3 = math.log(bi_temp3, 10)
            log_tri_count3 += log_tri_temp3
            log_bi_count3 += log_bi_temp3
    p0 = log_tri_count0 - log_bi_count0
    p1 = log_tri_count1 - log_bi_count1
    p2 = log_tri_count2 - log_bi_count2
    p3 = log_tri_count3 - log_bi_count3
    # print('pppp')
    print(p0, p1, p2, p3)
    psum = 10**p0+10**p1+10**p2+10**p3
    print(psum)
    # logp = math.log(psum, 10)
    # print(logp)
    # pp = pow(10, (-1*logp)/(len(test_sent)))
    # return pp
    logp = p0 + p1 + p2 + p3
    # logp = math.log(psum, 10)
    print(logp)
    pp = pow(10, (-1*psum)/(len(test_sent)))
    return pp

def perplex3(test_sent, mylist):
    tri_sent = list(trigrams(test_sent))
    ftri = freqdic(list(trigrams(mylist)))
    fbi = freqdic(list(bigrams(mylist)))
    fui = freqdic(mylist)
    log_tri_count = 0
    log_bi_count = 0
    v = number_v(mylist)
    for tri in tri_sent:
        c3 = ftri[tri]
        c2 = fbi[tri[:2]]
        c1 = fui[tri[0]]
        if ftri[tri] != 0:
            tri_temp = (c3 * 0.37)*c1*v*v + v*v*c2*c2 * 0.37+0.24*c1*c1*c2*v+ 0.02*c1*c2*v
            bi_temp = c1*c2*v*v
        elif ftri[tri] == 0 and fbi[tri[:2]] != 0:
            tri_temp = 0.37*c2*v*v+0.24*c1*c1*v+0.02*c1*v
            bi_temp = c1*v*v
        elif ftri[tri] == 0 and fbi[tri[:2]] == 0 and fui[tri[0]] != 0:
            tri_temp = 0.24*c1+0.02
            bi_temp = v
        elif fui[tri[0]] == 0:
            tri_temp = 1 * 0.02
            bi_temp = v
        log_tri_temp = math.log(tri_temp, 10)
        log_tri_count += log_tri_temp
        log_bi_temp = math.log(bi_temp, 10)
        log_bi_count += log_bi_temp
    p = log_tri_count - log_bi_count
    print('old perplexity ')
    print(p)
    log_pp = (-1*p) /(len(test_sent))
    pp = pow(10, log_pp)
    return pp

def perplex4(test_sent, mylist):
    tri_sent = list(trigrams(test_sent))
    ftri = freqdic(list(trigrams(mylist)))
    fbi = freqdic(list(bigrams(mylist)))
    fui = freqdic(mylist)
    log_tri_count = 0
    log_bi_count = 0
    v = number_v(mylist)
    a1 = 0.4
    a2 = 0.38
    a3 = 0.2
    a4 = 0.02
    for tri in tri_sent:
        c3 = ftri[tri]
        c2 = fbi[tri[:2]]
        c1 = fui[tri[0]]
        if ftri[tri] != 0:
            tri_temp = (c3 * a1)*c1*v*v + v*v*c2*c2 * a2+ a3*c1*c1*c2*v + a4*c1*c2*v
            bi_temp = c1*c2*v*v
        elif ftri[tri] == 0 and fbi[tri[:2]] != 0:
            tri_temp = a2*c2*v*v+ a3*c1*c1*v+ a4*c1*v
            bi_temp = c1*v*v
        elif ftri[tri] == 0 and fbi[tri[:2]] == 0 and fui[tri[0]] != 0:
            tri_temp = a3*c1 + a4
            bi_temp = v
        elif fui[tri[0]] == 0:
            tri_temp = a4
            bi_temp = v
        log_tri_temp = math.log(tri_temp, 10)
        log_tri_count += log_tri_temp
        log_bi_temp = math.log(bi_temp, 10)
        log_bi_count += log_bi_temp
    p = log_tri_count - log_bi_count
    # print('old perplexity ')
    # print(p)
    log_pp = (-1*p) /(len(test_sent))
    pp = pow(10, log_pp)
    return pp


if __name__ == '__main__':
    start = time.time()
    data_list_all = []
    data_without_lemma_16000 = []
    data_without_lemma_3000 = []
    p_num_all, n_num_all = 0, 0
    pos_sto, neg_sto = 0, 0
    count_line = 0
    pattern_no_url = re.compile(r'(https://|http://|www\.)+(\w|\.|/|\?|=|&|%)*\b')
    pattern_no_num = re.compile(r'[a-z0-9]*[a-z][a-z0-9]*')
    pattern_no_1letter = re.compile(r'[a-z0-9]{2,}')
    with open('signal-news1.jsonl', 'r') as f:
    # with open('mytest.jsonl', 'r') as f:
        for line in f.readlines():
            content = json.loads(line).get('content').lower()
            content_no_url = pattern_no_url.sub('', content)
            content_no_num = ' '.join(pattern_no_num.findall(content_no_url))
            content_no_1 = ' '.join(pattern_no_1letter.findall(content_no_num))
            content_nolemma_list = content_no_1.split()
            content_alldone_list = lemma(content_no_1)
            data_list_all.extend(content_alldone_list)
            if count_line < 16000:
            # if count_line < 15:
                data_without_lemma_16000.extend(content_nolemma_list)
            else:
                data_without_lemma_3000.extend(content_nolemma_list)
            count_line += 1
            p_n_num = count_p_n(content_alldone_list)
            p_num_all += p_n_num[0]
            n_num_all += p_n_num[1]
            p_n_dif = p_n_num[0] - p_n_num[1]
            if p_n_dif > 0:
                pos_sto += 1
            elif p_n_dif < 0:
                neg_sto += 1

    time1 = time.time()
    print('time A ' + str(time1 - start))
    print('N:%d' % len(data_list_all))
    print('V:%d' % number_v(data_list_all))
    tri_withcount_all = most_common(freqdic(list(trigrams(data_list_all))))[:25]
    tri_list_all = all_trigram(tri_withcount_all)
    print(tri_list_all)

    print('num of positive words in corpus: %d, negative words: %d' % (p_num_all, n_num_all))
    print('num of positive stories: %d, negative stories: %d' % (pos_sto, neg_sto))
    print('part B done')
    time2 = time.time()
    print('time B '+str(time2 - time1))

    tri_withcount_nolemma16000 = all_trigram_withcount(data_without_lemma_16000)
    tri_list_nolemma16000 = all_trigram(tri_withcount_nolemma16000)
    sen = gen_sentence(tri_list_nolemma16000, 'is', 'this', 10)
    print(sen)

    pp_back = perplex4(data_without_lemma_3000, data_without_lemma_16000)
    print('perplexity_back1 of remaining rows is ' + str(pp_back))
    # pp_back2 = perplex2(data_without_lemma_3000, data_without_lemma_16000)
    # print('perplexity_back122 of remaining rows is ' + str(pp_back2))

    pp_back2 = perplex3(data_without_lemma_3000, data_without_lemma_16000)
    print('perplexity_back122 of remaining rows is ' + str(pp_back2))
    end = time.time()
    time3 = time.time()
    print('time C ' + str(time3 - time2))
    print('cost time '+str(end-start))
