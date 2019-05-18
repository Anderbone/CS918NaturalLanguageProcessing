import json


# print the JSON fields
def explore(filename):
	with open(filename, 'r') as f:
		line = f.readline() # read only the first tweet/line
		tweet = json.loads(line)
		print(json.dumps(tweet, indent=8))
	f.close()


# print out the values of a specific field
def printStuff(filename, field):
	with open(filename, 'r') as f:
		for line in f:
			tweet = json.loads(line)
			print(tweet[field])
	f.close()


# print out the values of a user-specific field
def printUserStuff(filename):
	with open(filename, 'r') as f:
		for line in f:
			user = json.loads(line)["user"]
			name = user["name"]
			username = user["screen_name"]
			description = user["description"]
			location = user["location"]
			timezone = user["time_zone"]
			numFollowing = user["friends_count"]	
			numFollowers = user["followers_count"]
			print(username + '\t' + str(numFollowers) + '\t' + str(numFollowing))
	f.close()




