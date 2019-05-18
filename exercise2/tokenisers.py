from nltk.tokenize import word_tokenize
import twokenize

# white space using the split() method
def tokenise_ws(tweet):
	tokens = tweet.split()
	return tokens


# try the nltk word_tokeniser
def tokenise_nltk(tweet):
	tokens = word_tokenize(tweet)
	return tokens


# Owoputi et al, 2013
def tokenise_twok(tweet):
	tokens = twokenize.tokenize(tweet)
	return tokens


tweet = "@user this is AWESOME!!1Let's get the#party started!http://someurl.com"
ws = tokenise_ws(tweet)
print(ws)
nltk = tokenise_nltk(tweet)
print(nltk)
twok = tokenise_twok(tweet)
print(twok)

