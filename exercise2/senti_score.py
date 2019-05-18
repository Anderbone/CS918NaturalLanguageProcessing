from nltk.corpus import wordnet
import collections
import nltk
import numpy as np
def get_senti_score(split_corpus):
    def senti_file():
        senti_dic = collections.defaultdict(lambda: 0)
        with open('SentiWords_1.1.txt', 'r') as f:
        # with open('SentiWords_1.2.txt', 'r') as f:
            for line in f:
                fields = line.strip().split('\t')
                # words = [every_line.split('\t') for every_line in f]
                # print(words)
                senti_dic[fields[0]] = float(fields[1])
                # print(fields)
                # score = fields[1]
                # word = fields[0].split('#')[0]
                # tag = fields[0].split('#')[1]
                # print(word)
                # print(tag)
                # print(score)
            # return words
            # print(senti_dic)
            return senti_dic
    mydic = senti_file()
    all_socre = []
    for line in split_corpus:
        after_tag = nltk.pos_tag(line)
        # print(after_tag)
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
                after = word +str('#n')
                score = mydic[after]
                # print(score)
                # print(type(score))
            score_feature += score
        all_socre.append([score_feature])
    ans = np.array(all_socre)
    print(ans)
    return ans

        # print(after_tag_line)
    senti_file()


split_corpus = [['Felt', 'privileged', 'to', 'play', 'Foo', 'Fighters', 'songs', 'on', 'guitar', 'today', 'with', 'one', 'of', 'the', 'plectrums', 'from', 'the', 'gig', 'on', 'Saturday'], ['USERMENTION', 'Pakistan', 'may', 'be', 'an', 'Islamic', 'country', 'but', 'der', 'are', 'lot', 'true', 'Muslims', 'in', 'India', 'who', 'love', 'their', 'country', 'and', 'can', 'sacrifice', 'all', 'for', 'it'], ['Happy', 'Birthday', 'to', 'the', 'coolest', 'golfer', 'in', 'Bali', 'USERMENTION', 'joy', 'may', 'you', 'become', 'cooler', 'and', 'cooler', 'everyday', 'Stay', 'humble', 'little', 'sister', 'Xx'], ['USERMENTION', 'TMILLS', 'is', 'going', 'to', 'Tucson', 'But', 'the', '29th', 'and', 'it', 'on', 'Thursday', 'bad']]
get_senti_score(split_corpus)
