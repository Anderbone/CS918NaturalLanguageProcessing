def gen_sentence(data_list_all, a, b):
    w3 = trigram_model(data_list_all, a, b)
    w4 = trigram_model(data_list_all, b, w3)
    w5 = trigram_model(data_list_all, w3, w4)
    w6 = trigram_model(data_list_all, w4, w5)
    w7 = trigram_model(data_list_all, w5, w6)
    w8 = trigram_model(data_list_all, w6, w7)
    w9 = trigram_model(data_list_all, w7, w8)
    w10 = trigram_model(data_list_all, w8, w9)
    sentence = [a, b, w3, w4, w5, w6, w7, w8, w9, w10]
    return sentence

def find_25tri(mylist):
    """

    :param mylist:  a list after all text pre-processing in partA
    :return: a list of 25 trigrams based on the number of occurrences on the corpus
    """
    # trigram_list = list(trigrams(mylist))
    # top25 = Counter(trigram_list).most_common(25)
    return all_trigram(mylist)[:25]
    tri_25 = []
    # for i in top25:
    #     tri_25.append(list(i[0]))
    # return tri_25
    # return top25


# def get_biSentence(min,max,genre,sentence=''):
#     print "computing bigrams and generating random sentence:"
#     table=get_biTable(genre)
#     length=len(sentence)
#     if length==0:
#         sentence=random_next(table['.'])
#     sentence_tokens=nltk.word_tokenize(sentence)
#     last_word=sentence_tokens[-1]
#     for x in range(max):
#         generating=True
#         while (generating):
#             if last_word in table:
#                 next=random_next(table[last_word])
#             else:
#                 next=random.choice(table.keys())
#             generating=False
#             if (next=='.' and len<min):
#                 generating=True
#         sentence=sentence+' '+next
#         if next=='.':
#             return sentence
#         length+=1
#         last_word=next
#     return sentence+'.'

def trigram_model(mylist, a, b):
    for i in mylist:
        if i[0] == a and i[1] == b:
            return str(i[2])

def perplex(test_list, tri_model):
    perplexity = 1
    n = len(test_list)
    # bi_list = list(bigrams(test_list))
    tri_list = list(trigrams(test_list))
    for trigram in tri_list:
        perplexity = perplexity * tri_model[trigram]
    perplexity = pow(perplexity, -1/n)
    return perplexity

def tri_model(mylist):
    model = collections.defaultdict(lambda: 0.01)
    tri_count = all_trigram_withcount(mylist)
    bi_count = all_bigram_withcount(mylist)
    for trigram in tri_count:
        for bigram in bi_count:
            if bigram[0][0] == trigram[0][0] and bigram[0][1] == trigram[0][1]:
                denominator = bigram[1]
                model[trigram[0]] = trigram[1]/denominator
    return model

    # for bigram in bi_count:
#     if bigram[0][0] == trigram[] and i[0][1] == 'this':
#         print(i[1])
# print(data_list_all, file = ans)
# data_str =' '.join(data_list_all)
# print(data_str, file = ans)
