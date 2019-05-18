from nltk.tokenize import word_tokenize
import twokenize, re, json


#tokenise using nltk.tokenize word_tokenize
def nltk_tokenise(tweet):
	tokens = word_tokenize(tweet)
	tokenised = ''
	for token in tokens:
		tokenised += str(token.encode('utf-8')+' ')
	return tokenised.strip()


#replace URLs with "URLLINK"
def replaceURLs(tweet):
	return re.sub(r"http\S+", "URLLINK", tweet)


#replace user mentions with "USERMENTION"
def replaceUserMentions(tweet):
	result = re.sub("(@[A-Za-z0-9_]+)", "USERMENTION", tweet)
	return result


#replace all non-alphanumeric
def replaceRest(tweet):
	result = re.sub("[^a-zA-Z0-9]", " ", tweet)
	return re.sub(' +',' ', result)


def testit():
	with open("mytweets.json", 'r') as f:
		for line in f:
			text = json.loads(line)['text']
			newtext = nltk_tokenise(text).lower()
			newtext = replaceURLs(newtext)
			newtext = replaceUserMentions(newtext)
			newtext = replaceRest(newtext)
			print(text + '\n' + newtext + '\n')
	f.close()

