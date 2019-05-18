import re

pattern_no_num = re.compile(r'[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*')
pattern_no_1letter = re.compile(r'[A-Za-z0-9]{2,}')


# replace URLs with "URLLINK"
def replaceURLs(tweet):
	return re.sub(r"http\S+", "URLLINK", tweet)


# replace user mentions with "USERMENTION"
def replaceUserMentions(tweet):
	return re.sub("(@[A-Za-z0-9_]+)", "USERMENTION", tweet)


# replace happy and sad face with "HAPPYFACE","SADFACE"
def replaceface(tweet):
	tweet = re.sub(r"(\^\^)|(\^\.\^)|(\^\-\^)|(:\-\))|(:\))|(:o\))|(:\])|(:3)|(:c\))|(:>)|(=\])|(8\))|(=\))|(:\})|(:\^\))|(:D)|(C:)|(:\-D)|(:D)|(8D)|(xD)|(XD)|(=D)|(=3)|(<=3)|(<=8)|(;\-\))|(;\))|(\*\))|(;\])|(;D)|(:\-P)|(:P)|(XP)|(:\-p)|(:p)|(=p)|(xP)|(:\-b)|(:b)|(:\-O)|(:O)|(O_O)|(o_o)|(OwO)|(O\-O)|(0_o)|(O_o)|(O3O)|(o0o)|(;o_o;)|(o\.\.\.o)|(0w0)|(d:\-\))|(qB\-\))|(:\)\~)|(:\-\)>)|(:\-X)|(:X)|(:\-\#)|(:\#)|(:\-x)|(:x)|(:\-\*)|(:\*)|(>:\))|(>;\))|(>:\-\))|(B\))|(B\-\))|(8\))|(8\-\))|(<3)|(<333)|(=\^_\^=)|(=>\.>=)|(=<_<=)|(=>\.<=)|(\\,,/)|(\\m/)|(\\m/\\>\.</\\m/)|(\\o/)|(\\oo/)|(d'\-')|(d'_')|(d'\-'b)|(b'_'b)|(o/\\o)|(:u)|(3:00)|(=\]:\-\)=)|(d\^_\^b)|(d\-_\-b)|(\(\^_\^\))|(\(\^\-\^\))|(\(\^=\^\))|(\(\^\.\^\))|(\(\~_\^\))|(\(\^_\~\))|(\~\.\^)|(\^\.\~)|(\(\^o\^\))|(\(\^3\^\))|(d\(>w<\)b)|(\^///\^)|(>///<)|(>///>)|(o///o)|(\-///\-)|(=///=)|(\(V\)!_!\(V\))|(0v0)|(\\m/\*\.\*\\m/)|(\&_\&)|(0\-0)|(\(\\_/\))|(B\))|(X3)|(:3)|(O\|\-\|_)|(orz)|(/\\_/\\)|(/\\)|(_/\\_)|(_/\-\\_)|(/\\/\\)", "HAPPYFACE", tweet)
	result = re.sub(r"(\-\-!)|(\-\-)|(:\-\()|(:\()|(:c)|(:<)|(:\[)|(:\{)|(D:)|(D8)|(D;)|(D=)|(DX)|(v\.v)|(Dx)|(:\-9)|(c\.c)|(C\.C)|(:\-/)|(:/)|(:\\)|(=/)|(=\\)|(:S)|(:\|)|(O:\-\))|(0:3)|(O:\))|(:'\()|(;\*\()|(T_T)|(TT_TT)|(T\.T)|(Q\.Q)|(Q_Q)|(;_;)|(\^o\))|(\^>\.>\^)|(\^<\.<\^)|(\^>_>\^)|(\^<_<\^)|(D:<)|(>:\()|(D\-:>)|(>:\-\()|(:\-@\[1\])|(;\()|(`_')|(D<)|(:\&)|(\(>\.<\))|(\(>\.<\))|(\(>_>\))|(\(<_<\))|(\(\-_\-\))|(\(\^_\^'\))|(9\(x\.x\)9)|(\(;\.;\)9)|(\(\._\.\))|(\(,\.,\))|(\(\-_\-\))|(Zzz)|(\(X_X\))|(x_x)|(_\|_)|(\(\-\.\-\))|(_\|_)|(\(t>\.<t\))|(O\?O)|(\(\^\^\^\))|(x_O)|(O_x)", "SADFACE", tweet)
	return result

# replace too long words like'Goooood' to 'God'
def replaceLong(tweet):
	return re.sub(r'([a-zA-Z])(\1){2,}', r'\1', tweet)


# replace 'lol' to 'HAPPYFACE'
def replacelol(tweet):
	return re.sub(r'\b(lol)\b', 'HAPPYFACE', tweet)


# replace pure number and 1-length character, also remove all non-alphanumeric
def replaceNum_1letter(tweet):
	tweet = ' '.join(pattern_no_num.findall(tweet))
	result = ' '.join(pattern_no_1letter.findall(tweet))
	return result


def replaceall(tweet):
	tweet = replaceface(tweet)
	tweet = replaceURLs(tweet)
	tweet = replaceUserMentions(tweet)
	tweet = replaceLong(tweet)
	tweet = replacelol(tweet)
	tweet = replaceNum_1letter(tweet)
	return tweet
