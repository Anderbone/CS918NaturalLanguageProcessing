import time
import random
import re
from nltk.tokenize import word_tokenize
import json
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

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
            # print(lemma)
        else:
            after_lemma = lemmatizer.lemmatize(word, pos=wntag)
            data_list.append(after_lemma)
            # print(lemma)
    return data_list

def lemma2():
    lemmatizer = WordNetLemmatizer()
    a=lemmatizer.lemmatize('is')
    print(a)

def lemma3(str):
    data_list = []
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(str.split())
    for word, tag in tagged:
        if tag.startswith('J'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.ADJ)
        elif tag.startswith('V'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)
        elif tag.startswith('N'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
        elif tag.startswith('R'):
            after_lemma = lemmatizer.lemmatize(word, pos=wordnet.ADV)
        else:
            after_lemma = word
        data_list.append(after_lemma)
    # return " ".join(data_list)
    return data_list


if __name__ == '__main__':
    start = time.time()
    data_list_all = []
    pos_sto, neg_sto = 0, 0
    with open('testline.jsonl', 'r', encoding='utf-8') as f:
    # with open('testtwo.jsonl', 'r', encoding='utf-8') as f:
    # with open('signal-news1.jsonl', 'r', encoding='utf-8') as f:
    # with open('mytest.jsonl', 'r') as f:
        for line in f.readlines():
                content = json.loads(line).get('content').lower()
                content_no_url =''.join(re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', content))
                content_no_num = ' '.join((re.findall(r'[a-z0-9]*[a-z][a-z0-9]*', content_no_url)))
                content_less = ' '.join(re.findall(r'[a-z0-9]{2,}', content_no_num))
                # content_alldone = lemma(content_less)
                content_alldone = lemma3(content_less)
                data_list_all.extend(content_alldone)
    print(data_list_all)

    print(len(data_list_all))
    end = time.time()
    print('cost time is '+str(end - start))