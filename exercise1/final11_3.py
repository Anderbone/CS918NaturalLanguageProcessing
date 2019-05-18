import re
import json
import nltk
import math
import time
import collections
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import bigrams, trigrams
# after no num, finally lemma.
# my own lemma function
# http revised
# re.compile revised

def number_n(mylist):
    return len(mylist)


def number_v(mylist):
    return len((set(mylist)))


def lemma(str):
    data_list = []
    lemmatizer = WordNetLemmatizer()
    after_tag = nltk.pos_tag(str.split())
    for word, tag in after_tag:
        if tag.startswith('J'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.ADJ)
        elif tag.startswith('V'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)
        elif tag.startswith('R'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.ADV)
        elif tag.startswith('N'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
        else:
            after_lemma = word
        data_list.append(after_lemma)
    return data_list


def freqdic(mylist):
    mydict = collections.defaultdict(lambda: 0)
    for key in mylist:
        # mydict[key] = mydict.get(key, 0) + 1
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


def all_trigram(trigram_withcount):  # the output is all trigram sorted as frequency and no duplicate
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

def perplex_back(test_sent, mylist):
    tri_sent = list(trigrams(test_sent))
    fui = freqdic(mylist)
    # fbi = freqdic(list(bigrams(mylist))[:-1])
    fbi = freqdic(list(bigrams(mylist)))
    ftri = freqdic(list(trigrams(mylist)))
    log_bi_count = 0
    log_tri_count = 0
    vocabulary = number_v(mylist)
    for tri in tri_sent:
        if ftri[tri] != 0:
            tri_temp = ftri[tri]*0.37
            bi_temp = fbi[tri[:2]]
        elif ftri[tri] == 0 and fbi[tri[:2]] != 0:
            # print('back to bigram')
            tri_temp = fbi[tri[:2]]*0.37
            bi_temp = fui[tri[0]]
        elif ftri[tri] == 0 and fbi[tri[:2]] == 0 and fui[tri[0]] != 0:
            # print('back to unigram')
            tri_temp = fui[tri[0]]*0.24
            bi_temp = 1
        elif fui[tri[0]] == 0:
            # print('unk unknown word')
            tri_temp = 0.02*1/vocabulary
            bi_temp = 1
        log_tri_temp = math.log(tri_temp, 10)
        log_tri_count += log_tri_temp
        # bi_temp = fbi[bi]
        log_bi_temp = math.log(bi_temp, 10)
        log_bi_count += log_bi_temp
    p = log_tri_count - log_bi_count
    log_pp = -1/(len(test_sent)) * p
    pp = pow(10, log_pp)
    return pp


if __name__ == '__main__':
    start = time.time()
    data_list_all = []
    data_without_lemma_16000 = []
    data_without_lemma_3000 = []
    data_16 = []
    data_3 = []
    p_num_all, n_num_all = 0, 0
    pos_sto, neg_sto = 0, 0
    count_line = 0
    pattern_no_url = re.compile(r'(https://|http://|www\.)+(\w|\.|/|\?|=|&|%)*\b')
    pattern_no_num = re.compile(r'[a-z0-9]*[a-z][a-z0-9]*')
    pattern_no_1letter = re.compile(r'[a-z0-9]{2,}')
    with open('signal-news1.jsonl', 'r') as f:
    # with open('mytest.jsonl', 'r') as f:
    # with open('testline.jsonl', 'r') as f:
        for line in f.readlines():
            content = json.loads(line).get('content').lower()
            # content_no_url = ''.join(re.sub(r'(https|http)?:?//?(\w|\.|/|\?|=|&|%)*\b', '', content))
            # content_no_url = ''.join(re.sub(r'(https\:\/\/|http\:\/\/|www\.)+(\w|\.|\/|\?|=|&|%)*\b', '', content))
            # content_no_url = ''.join(re.sub(r'(https://|http://|www\.)+(\w|\.|/|\?|=|&|%)*\b', '', content))
            content_no_url = pattern_no_url.sub('', content)
            # content_no_url = ''.join(re.sub(r'[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+\.?', '', content))
            content_no_num = ' '.join(pattern_no_num.findall(content_no_url))
            # content_no_num = ' '.join((re.findall(r'[a-z0-9]*[a-z][a-z0-9]*', content_no_url)))
            # content_no_1 = ' '.join(re.findall(r'[a-z0-9]{2,}', content_no_num))
            # print(content_no_num)
            content_no_1 = ' '.join(pattern_no_1letter.findall(content_no_num))
            content_nolemma_list = content_no_1.split()

            content_alldone_list = lemma(content_no_1)
            data_list_all.extend(content_alldone_list)
            if count_line < 16000:
            # if count_line < 15:
            # if count_line < 1:
            #     data_16.extend(content_alldone_list)
                data_without_lemma_16000.extend(content_nolemma_list)
            else:
                # data_3.extend(content_alldone_list)
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

    # Above is part A
    # part B
    # for i in data_list_all:
    #     if len(i) == 1:
    #         print('wrong! one! why!')
    # print(data_list_all)
    # data_list_all2 = data_16 + data_3
    time1 = time.time()
    print('time A ' + str(time1 - start))
    print('N:%d' % number_n(data_list_all))
    print('V:%d' % number_v(data_list_all))
    tri_withcount_all = most_common(freqdic(list(trigrams(data_list_all))))[:25]
    tri_list_all = all_trigram(tri_withcount_all)
    print(tri_list_all)

    print('num of positive words in corpus: %d, negative words: %d' % (p_num_all, n_num_all))
    print('num of positive stories: %d, negative stories: %d' % (pos_sto, neg_sto))
    print('part B done')
    time2 = time.time()
    print('time B '+str(time2 - time1))

    # partC
    tri_withcount_nolemma16000 = all_trigram_withcount(data_without_lemma_16000)
    tri_list_nolemma16000 = all_trigram(tri_withcount_nolemma16000)
    sen = gen_sentence(tri_list_nolemma16000, 'is', 'this', 10)
    print(sen)

    # tri_withcount_lemma = all_trigram_withcount(data_16)
    # tri_list_lemma = all_trigram(tri_withcount_lemma)
    # sen2 = gen_sentence(tri_list_lemma, 'be', 'this', 10)
    # print(sen2)

    pp_back = perplex_back(data_without_lemma_3000, data_without_lemma_16000)
    print('perplexity_back1 of remaining rows is ' + str(pp_back))

    # pp_lemma = perplex_back(data_3, data_16)
    # print('perplexity_back2 of remaining rows is ' + str(pp_lemma))
    end = time.time()
    time3 = time.time()
    print('time C ' + str(time3 - time2))
    print('cost time '+str(end-start))
