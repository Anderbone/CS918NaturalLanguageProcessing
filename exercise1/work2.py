import re
import json
import nltk
import time
import random
import math
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams
from collections import Counter

import collections
from nltk import text
# from nltk.collocations import TrigramAssocMeasures,TrigramCollocationFinder


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    # elif treebank_tag.startswith('R'):
    #     return wordnet.ADV
    else:
        return None


def lemma(str):
    data_list = []
    lemmatizer = WordNetLemmatizer()
    # print(lemmatizer.lemmatize('going', wordnet.VERB))
    tagged = nltk.pos_tag(word_tokenize(str))
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # not supply tag in case of None
            after_lemma = lemmatizer.lemmatize(word)
            data_list.append(after_lemma)
            # print(lemma)
        else:
            after_lemma = lemmatizer.lemmatize(word, pos=wntag)
            data_list.append(after_lemma)
            # print(lemma)
    return data_list


def number_n(mylist):
    return len(mylist)


def number_v(mylist):
    return len((set(mylist)))


def all_trigram_withcount(mylist):   # Use Counter to sort all trigram.
    trigram_withcount = Counter(list(trigrams(mylist))).most_common()
    return trigram_withcount

def all_bigram_withcount(mylist):
    bigram_withcount = Counter(list(bigrams(mylist))).most_common()
    # for i in bigram_withcount:
    #     # print(i)
    #     if i[0][0] == 'be'and i[0][1] == 'this':
    #         print(i[1])
    return bigram_withcount

def all_trigram(trigram_withcount):
    trigram_list = []
    for i in trigram_withcount:
        trigram_list.append(i[0])
    return trigram_list


def gen_sentence(tri_list, a, b, leng):
    sen = [a, b]
    sen_go = True

    def gen_sentence_in(tri_list, a, b, leng):
        # global sen
        # global sen_go
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
    bi_sent = list(bigrams(test_sent))[:-1]
    bi_list = list(bigrams(mylist))[:-1]
    tri_sent = list(trigrams(test_sent))
    tri_list = list(trigrams(mylist))
    tri_count = 1
    bi_count = 1
    # model = collections.defaultdict(lambda: 0.01)
    # tri_count = all_trigram_withcount(mylist)
    # bi_count = all_bigram_withcount(mylist)
    for tri in tri_sent:
        if tri in tri_list:
            tri_count *= tri_list.count(tri)
        else:
            print("need tri smooth")
            tri_count *= 0.01
    for bi in bi_sent:
        if bi in bi_list:
            bi_count *= bi_list.count(bi)
        else:
            print('need bi smooth')
            bi_count *= 1
    pp = pow(tri_count/bi_count, -1/len(test_sent))
    return pp


# Add 1 smooth
def perplex3(test_sent, mylist):
    bi_sent = list(bigrams(test_sent))[:-1]
    bi_list = list(bigrams(mylist))[:-1]
    tri_sent = list(trigrams(test_sent))
    tri_list = list(trigrams(mylist))
    tri_count = 1
    bi_count = 1
    log_bi_count = 0
    log_tri_count = 0
    every_p = 0
    # model = collections.defaultdict(lambda: 0.01)
    # tri_count = all_trigram_withcount(mylist)
    # bi_count = all_bigram_withcount(mylist)
    for tri in tri_sent:
        tri_temp = tri_list.count(tri) + 1
        log_tri_temp = math.log(tri_temp, 10)
        # tri_count *= tri_temp
        log_tri_count += log_tri_temp
    for bi in bi_sent:
        vocabulary = number_v(mylist)
        bi_temp = bi_list.count(bi) + vocabulary
        log_bi_temp = math.log(bi_temp, 10)
        log_bi_count += log_bi_temp

    # print(tri_count)
    # print(bi_count)
    print(len(test_sent))
    every_p = log_tri_count - log_bi_count
    print(every_p)
    log_pp = -1/(len(test_sent)) * every_p
    print(log_pp)
    # pp = pow(tri_count/bi_count, -1/(len(test_sent)))
    return log_pp

def inputfile(file):
    with open(file, 'r') as my_file:
        words = [every_line.rstrip() for every_line in my_file]
        return words


def count_p_n(mylist):
    pos_num = 0
    neg_num = 0
    positive = inputfile('positive-words.txt')
    negative = inputfile('negative-words.txt')
    for word in mylist:
        if word in positive:
            pos_num += 1
        if word in negative:
            neg_num += 1
    return pos_num, neg_num


if __name__ == '__main__':
    start = time.time()
    data_list_16000 = []
    data_list_3000 = []
    data_without_lemma_16000 = []
    data_without_lemma_3000 = []
    line_without_lemma_3000 = []
    # data_list_perp = []
    p_num_all, n_num_all = 0, 0
    pos_sto, neg_sto = 0, 0
    count_line = 0
    # with open('testline.jsonl', 'r', encoding='utf-8') as f:
    # with open('testtwo.jsonl', 'r', encoding='utf-8') as f:
    with open('mytest.jsonl', 'r') as f:
    # with open('signal-news1.jsonl', 'r', encoding='utf-8') as f:
        for line in f.readlines():
                content = json.loads(line).get('content').lower()
                content_no_url =''.join(re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', content))
                # if count_line >= 16000:
                #     content_no_num_withdot = ' '.join((re.findall(r'\.?[a-z0-9]*[a-z][a-z0-9]*\.?', content_no_url)))
                #     content_less_withdot = ' '.join(re.findall(r'\.?[a-z0-9]{2,}\.?', content_no_num_withdot))
                #     sent_without_lemma_3000.append(content_less_withdot.split('.'))
                # content_less = ' '.join(re.findall(r'[a-z0-9]{2,}', content_no_url))
                # content_no_num = ' '.join((re.findall(r'[a-z0-9]*[a-z][a-z0-9]*', content_less)))
                content_no_num = ' '.join((re.findall(r'[a-z0-9]*[a-z][a-z0-9]*', content_no_url)))
                content_less = ' '.join(re.findall(r'[a-z0-9]{2,}', content_no_num))
                # print(content_less)
                # print(content_less.split())
                content_alldone = lemma(content_less)
                # content_alldone = lemma(content_no_num)
                # print(content_alldone)
                # if count_line < 16000:
                if count_line < 15:
                    data_without_lemma_16000.extend(content_less.split())
                    data_list_16000.extend(content_alldone)
                else:
                    # print(content_less)
                    line_without_lemma_3000.append(content_less)
                    data_without_lemma_3000.extend(content_less.split())
                    data_list_3000.extend(content_alldone)
                    # data_list_perp.append(content_alldone)
                count_line += 1
                p_n_num = count_p_n(content_alldone)
                p_num_all += p_n_num[0]
                n_num_all += p_n_num[1]
                p_n_dif = p_n_num[0] - p_n_num[1]
                # p_n = (count_p_n(content_alldone)[0]-count_p_n(content_alldone)[1])
                if p_n_dif > 0:
                    pos_sto += 1
                elif p_n_dif < 0:
                    neg_sto += 1

            # else:
            #     break
    # text = nltk.Text(''.join(data_list_all))
    # print(text.generate(text))


    data_list_all = data_list_16000 + data_list_3000
    print('N:%d' % number_n(data_list_all))
    print('V:%d' % number_v(data_list_all))
    # print(data_list_all)

    # print(type(data_list_all))
    # print(len(data_list_all))

    tri_withcount_all = all_trigram_withcount(data_list_all)
    tri_withcount_16000 = all_trigram_withcount(data_list_16000)
    tri_withcount_nolemma16000 = all_trigram_withcount(data_without_lemma_16000)
    # tri_withcount_remaining = all_trigram_withcount(data_list_remaining)

    tri_list_all = all_trigram(tri_withcount_all)
    tri_list_16000 = all_trigram(tri_withcount_16000)
    tri_list_nolemma16000 = all_trigram(tri_withcount_nolemma16000)
    # tri_list_remaining = all_trigram(tri_withcount_remaining)
    print(tri_list_all[0:25])

        # tri_list_remaining = all_trigram(all_trigram_withcount(data_list_remaining))
        # sen = gen_sentence(tri_list, 'share', 'of', 5)
        # sen = gen_sentence(tri_list, 'hello', 'how')
        # print(gen_sentence(tri_list_16000, 'hello', 'how', 10))

    # print('num of positive words in corpus: %d, negative words: %d' % (p_num_all, n_num_all))
    # print('num of positive stories: %d, negative stories: %d' % (pos_sto, neg_sto))

    # sen = gen_sentence(tri_list_16000, 'be', 'this', 10)
    # sen = gen_sentence(tri_list_16000, 'on', 'share', 10)
    # sen_nolemma = gen_sentence(tri_list_nolemma16000, 'is', 'this', 10)
    sen_nolemma = gen_sentence(tri_list_nolemma16000, 'last', 'year', 10)
    # sen = gen_sentence(tri_list_16000, 'be', 'this', 10)
    # print(sen)
    print(sen_nolemma)
    # ans = gen_sentence(tri_list, 'share', 'of', 10)
    # print(ans)
    pp_sum = 0
    for line in line_without_lemma_3000:
        pp = perplex3(line.split(), data_without_lemma_16000)
        print(pp)
        pp_sum += pp
    print('sum of perplexity of 3000 remaining rows is '+ str(pp_sum))
    # data_test_list_all = ['what', 'an', 'amazing', 'thing']
    # bi = bigrams(data_test_list_all)
    # tri = trigrams(data_test_list_all)
    # bi_withcount_all = all_bigram_withcount(data_list_all)
    # print(data_list_test)
    # model = tri_model(data_list_3000)
    # pp = perplex(sen, model)
    # pp = perplex2(sen, data_list_3000)
    # print('the final perplexity is ' + str(pp))
    # p_all = 0
    # for every_line in sent_without_lemma_3000:
    #     # print(every_line)
    #     print(every_line)
    #     for every_sen in every_line:
    #     # every_sen = every_line.split(',')
    #     #     print(every_sen)
    #         every_pp = perplex3(every_sen.split(), data_without_lemma_16000)
    #     # perplex3(every_line.split(), data_without_lemma_16000)
    #         print(every_pp)
    #         p_all += every_pp
    # print('3000 no lem  ma perplex is ' + str(p_all))
    # model = tri_model(data_list_16000)
    # for line in data_list_perp:
    #     pp = perplex(line, model)
    #     print('the final perplexity ' + str(pp), file=ans)


end = time.time()
print('cost time is '+str(end - start))
