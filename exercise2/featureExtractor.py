from sklearn.feature_extraction.text import CountVectorizer

# returns matrix
# one row per document
# one column per unigram
# Source: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def getUnigrams():
	vectorizer = CountVectorizer()
	corpus = ['this is a document.',
	'another document.',
	'yet another document!']
	X = vectorizer.fit_transform(corpus)
	print(vectorizer.vocabulary_)
	print(X.todense())




