import word2vecReader
from gensim.models import word2vec
import numpy as np
# newmodel = word2vecReader.Word2Vec()

def word_embedding3(split_corpus):
    # print('word embedding3 this is my new divided version --------------------')
    all = []
    for i in split_corpus:
        # print(i)
        # print(len(i))
        model = word2vec.Word2Vec([i], min_count=0, size=8)
        # print(model.vocabulary)
        # print(len(model.vocabulary))
        # print(model.wv.vocab)
        # s = model.wv.syn0
        s0 = model.wv.vectors
        # print(len(s0))
        # print(s0)
        s = [x / len(s0) for x in s0]
        # print('before')
        ans = list(map(sum, zip(*s)))  # sum of them.  All words in one tweet.
        # print('after')
        # print(len(ans))
        all.append(ans)
    arr = np.array(all)
    # return np.array(all)
    # print(len(arr))
    # print(arr)
    return arr

model_path = "word2vec_twitter_model.bin"
model = word2vecReader.Word2Vec.load_word2vec_format(model_path, binary=True)
my = [['Felt', 'privileged', 'to', 'play', 'Foo', 'Fighters', 'songs', 'on', 'guitar', 'today', 'with', 'one', 'of', 'the', 'plectrums', 'from', 'the', 'gig', 'on', 'Saturday'], ['USERMENTION', 'Pakistan', 'may', 'be', 'an', 'Islamic', 'country', 'but', 'der', 'are', 'lot', 'true', 'Muslims', 'in', 'India', 'who', 'love', 'their', 'country', 'and', 'can', 'sacrifice', 'all', 'for', 'it'], ['Happy', 'Birthday', 'to', 'the', 'coolest', 'golfer', 'in', 'Bali', 'USERMENTION', 'joy', 'may', 'you', 'become', 'cooler', 'and', 'cooler', 'everyday', 'Stay', 'humble', 'little', 'sister', 'Xx'], ['USERMENTION', 'TMILLS', 'is', 'going', 'to', 'Tucson', 'But', 'the', '29th', 'and', 'it', 'on', 'Thursday', 'bad']]

# for sentence in my:
#     vector = np.zeros(400)
#     d = 0
#     for word in sentence:
#         if word in model.wv.vocab:
#             vector += model.syn0norm[model.vocab[word].index]
#             d += 1
#         vector = vector / d
#         print(vector)
mysen = ['Felt', 'privileged', 'to', 'play', 'Foo']

def word_embedding4(split_corpus, model=model, size=400):
    # using external twitter specific per-trained
    # words = preprocess(text)
    ans = []
    for line in split_corpus:
        vec = np.zeros(size)
        count = 0.
        for word in line:
            try:
                vec += model[word]
                count += 1.
            except KeyError:
                continue
        if count != 0:
            print(count)
            vec /= count
        # print(len(vec))
        print(vec)
        print(type(vec))
        # ans = np.append(ans, vec.tolist())
        ans.append(vec.tolist())
        # ans.append(vec)
        # ans = np.array(ans)
    # print(len(ans))
    # print(ans)
    arr = np.array(ans)
    print(arr)
    print(len(arr))
    return arr
# word_embedding2(my)
get_vector(my)
# X_pos = open('data/positive-all','r').readlines()
# X_neu = open('data/neutral-all','r').readlines()
# X_neg = open('data/negative-all','r').readlines()
#
#
# X_pos_vec = np.array(map(get_vector, X_pos))
# X_neu_vec = np.array(map(get_vector, X_neu))
# X_neg_vec = np.array(map(get_vector, X_neg))
#
#
# y_pos_vec = np.ones(len(X_pos_vec))
# y_neu_vec = np.zeros(len(X_neu_vec))
# y_neg_vec = np.full(len(X_neg_vec),-1)
#
# X_all = np.concatenate((X_pos_vec, X_neu_vec, X_neg_vec))
# y_all = np.concatenate((y_pos_vec, y_neu_vec, y_neg_vec))
#
# X_all = scale(X_all)
#
# X_all.dump('models/X_all_w2v')
# y_all.dump('models/y_all_w2v')
#
#
# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.4, random_state=42)
#
# logit = LogisticRegression(C=0.5)
# clf = logit.fit(X_train, y_train)
# pred = clf.predict(X_test)
# print classification_report(y_test, pred, target_names=['1.','0.','-1.'])

