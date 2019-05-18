from nltk.probability import FreqDist
from nltk import trigrams
import collections

mylist = ['happy','happy','not','what','happy','happy','not']


def Freqdic(mylist):
    mydict = dict()
    for key in mylist:
        mydict[key] = mydict.get(key, 0) + 1
    return mydict

def most_common(mydict):
    return sorted(mydict.items(), key=lambda d: d[1])


def freqdic(mylist):
    mydict = collections.defaultdict(lambda: 0)
    for key in mylist:
        mydict[key] += 1
    return mydict



tri_sent = list(trigrams(mylist))
fui = FreqDist(mylist)
fti = FreqDist(tri_sent)
for ft in fti:
    print(ft,fti[ft])

trigram_withcount = fti.most_common()
print(trigram_withcount)
print('----')
most_common(freqdic(tri_sent))

print(trigram_withcount)
fti2 = Freqdic(tri_sent)
for i in fti2:
    print(i,fti2[i])
for ui in fui:
    print(ui,fui[ui])
# for tri in tri_sent:
#     print(tri)
#     print(fui[tri[0]])