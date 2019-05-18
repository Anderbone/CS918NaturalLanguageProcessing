import re

import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


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


content = 'abc e 555dd aaff12 apple going abc_fdd fighting apples sadly pleasantly http://www.badi.com *dsaa/df$$&  6666'
print(content)

content_no_url = ''.join(re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', content, flags=re.MULTILINE))
print(content_no_url)

content_less =' '.join(re.findall(r'[a-z0-9]{4,}',content_no_url))
print(content_less)

# content_a = ''.join(re.findall(r'[a-z0-9\s]', content_less)).replace("\n", "")
# print(content_a)

content_no_num = ' '.join((re.findall(r'[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*', content_less)))
print(content_no_num)
print(type(content_no_num))

lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('going', wordnet.VERB))
tagged = nltk.pos_tag(word_tokenize(content_no_num))
for word, tag in tagged:
    wntag = get_wordnet_pos(tag)
    if wntag is None:# not supply tag in case of None
        lemma = lemmatizer.lemmatize(word)
        print(lemma)
    else:
        lemma = lemmatizer.lemmatize(word, pos=wntag)
        print(lemma)
