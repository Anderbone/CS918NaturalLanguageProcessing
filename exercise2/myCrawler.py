import tweepy, json
import os, sys, getopt


class CustomStreamListener(tweepy.StreamListener):
	_filename = 'mytweets.json'
	_ffile = open(_filename, 'a')

	def on_status(self, status):
        	CustomStreamListener._ffile.write(str(json.dumps(status._json))+'\n')

	def on_error(self, status_code):
		print(sys.stderr, 'Error:', status_code)
		return True

	def on_timeout(self):
		print(sys.stderr, 'Timeout...')
		return True


def getStreamer():
	consumer_key="CK"
	consumer_secret="CKS"
	access_key="123-AK"
	access_secret="AS"
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	return tweepy.streaming.Stream(auth, CustomStreamListener())


def crawl(streamerType):
	strapi = getStreamer()
	if streamerType == 'keyword':
		keywords = getKeywords()
		strapi.filter(track=keywords)
	elif streamerType == 'location':
		locs = getLocations()
		strapi.filter(locations=locs)
	elif streamerType == 'follow':
		accounts = getAccountsToFollow()
		strapi.filter(follow=accounts)


# define a set of keywords to look up for
def getKeywords():
	return ['java', 'python']

# define a list of accounts to follow
def getAccountsToFollow():
	return ['@BBC', '@guardian']

# define some locations to look up for
def getLocations():
	return	[-74.255735, 40.496044, -73.700272, 40.915256,	#NY
		-118.668176, 33.703692, -118.155289, 34.337306]	#LA


def getParams(argv):
	streamerType = ''
	try:
		opts, args = getopt.getopt(argv,"hi:",["ifile="])
	except getopt.GetoptError:
 		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			sys.exit()
		elif opt in ("-i", "--ifile"):
			streamerType = arg
	return streamerType


if __name__ == "__main__":
	crawl(getParams(sys.argv[1:]))

