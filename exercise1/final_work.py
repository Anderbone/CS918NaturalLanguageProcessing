import re
import json
import nltk
import time
import math
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams
from collections import Counter


def number_n(mylist):
    return len(mylist)


def number_v(mylist):
    return len((set(mylist)))


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemma(str):
    data_list = []
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(word_tokenize(str))
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # not supply tag in case of None
            after_lemma = lemmatizer.lemmatize(word)
            data_list.append(after_lemma)
        else:
            after_lemma = lemmatizer.lemmatize(word, pos=wntag)
            data_list.append(after_lemma)
    return data_list


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


def all_trigram_withcount(mylist):   # Use Counter to sort all trigram.
    trigram_withcount = Counter(list(trigrams(mylist))).most_common()
    return trigram_withcount


def all_bigram_withcount(mylist):
    bigram_withcount = Counter(list(bigrams(mylist))).most_common()
    return bigram_withcount


def all_trigram(trigram_withcount):  # the output is all trigram sorted as count and no duplicate
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
    bi_sent = list(bigrams(test_sent))[:-1]
    tri_sent = list(trigrams(test_sent))
    bi_list = list(bigrams(mylist))[:-1]
    tri_list = list(trigrams(mylist))
    log_bi_count = 0
    log_tri_count = 0
    for tri in tri_sent:
        tri_temp = tri_list.count(tri) + 1
        log_tri_temp = math.log(tri_temp, 10)
        log_tri_count += log_tri_temp
    for bi in bi_sent:
        vocabulary = number_v(mylist)
        bi_temp = bi_list.count(bi) + vocabulary
        log_bi_temp = math.log(bi_temp, 10)
        log_bi_count += log_bi_temp
    every_p = log_tri_count - log_bi_count
    log_pp = -1/(len(test_sent)) * every_p
    return log_pp


if __name__ == '__main__':
    start = time.time()
    data_list_16000 = []
    data_list_3000 = []
    data_without_lemma_16000 = []
    data_without_lemma_3000 = []
    p_num_all, n_num_all = 0, 0
    pos_sto, neg_sto = 0, 0
    count_line = 0
    with open('testline.jsonl', 'r', encoding='utf-8') as f:
    # with open('testtwo.jsonl', 'r', encoding='utf-8') as f:
    # with open('mytest.jsonl', 'r') as f:
    # with open('signal-news1.jsonl', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = json.loads(line).get('content').lower()
            content_no_url = ''.join(re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', content))
            content_no_num = ' '.join((re.findall(r'[a-z0-9]*[a-z][a-z0-9]*', content_no_url)))
            content_less = ' '.join(re.findall(r'[a-z0-9]{2,}', content_no_num))
            content_alldone = lemma(content_less)
            if count_line < 1:
            # if count_line < 15:
                data_without_lemma_16000.extend(content_less.split())
                data_list_16000.extend(content_alldone)
            else:
                # line_without_lemma_3000.append(content_less)
                data_without_lemma_3000.extend(content_less.split())
                data_list_3000.extend(content_alldone)
            count_line += 1
            p_n_num = count_p_n(content_alldone)
            p_num_all += p_n_num[0]
            n_num_all += p_n_num[1]
            p_n_dif = p_n_num[0] - p_n_num[1]
            if p_n_dif > 0:
                pos_sto += 1
            elif p_n_dif < 0:
                neg_sto += 1

    data_list_all = data_list_16000 + data_list_3000
    print('N:%d' % number_n(data_list_all))
    print('V:%d' % number_v(data_list_all))

    tri_withcount_all = all_trigram_withcount(data_list_all)
    tri_withcount_nolemma16000 = all_trigram_withcount(data_without_lemma_16000)

    tri_list_all = all_trigram(tri_withcount_all)
    tri_list_nolemma16000 = all_trigram(tri_withcount_nolemma16000)
    print(tri_list_all[0:25])

    print('num of positive words in corpus: %d, negative words: %d' % (p_num_all, n_num_all))
    print('num of positive stories: %d, negative stories: %d' % (pos_sto, neg_sto))

    # sen = gen_sentence(tri_list_nolemma16000, 'last', 'year', 10)
    sen = gen_sentence(tri_list_nolemma16000, 'is', 'this', 10)
    print(sen)
    # pp_sum = 0
    # for line in line_without_lemma_3000:
    pp = perplex(data_without_lemma_3000, data_without_lemma_16000)
    #     print('every line pp' + str(pp))
    #     pp_sum += pp
    print('sum of perplexity of 3000 remaining rows is ' + str(pp))
    end = time.time()
    print('cost time is ' + str(end - start))
