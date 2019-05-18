from nltk.tokenize import word_tokenize
import twokenize, re, json
import nltk


#tokenise using nltk.tokenize word_tokenize
# using the twokenize.py, not this one.
def nltk_tokenise(tweet):
	tokens = word_tokenize(tweet)
	tokenised = ''
	for token in tokens:
		tokenised += str(token.encode('utf-8')+' ')
	return tokenised.strip()

pattern_no_num = re.compile(r'[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*')
pattern_no_1letter = re.compile(r'[A-Za-z0-9]{2,}')

#replace URLs with "URLLINK"
def replaceURLs(tweet):
	return re.sub(r"http\S+", "URLLINK", tweet)

Plist = [':-)', ':)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ';-)', ';)', '*)', ';]', ';D', ':-P', ':P', 'XP', ':-p', ':p', '=p', 'xP', ':-b', ':b', ':-O', ':O', 'O_O', 'o_o', 'OwO', 'O-O', '0_o', 'O_o', 'O3O', 'o0o', ';o_o;', 'o...o', '0w0', 'd:-)', 'qB-)', ':)~', ':-)>....', ':-X', ':X', ':-#', ':#', ':-x', ':x', ':-*', ':*', '>:)', '>;)', '>:-)', 'B)', 'B-)', '8)', '8-)', '<3', '<333', '=^_^=', '=>.>=', '=<_<=', '=>.<=', '\\o', 'o/', "d'-'", "d'_'", "d'-'b", "b'_'b", 'o/\\o', ':u', "@}-;-'---", '3:00', '=]:-)=', 'd^_^b', 'd-_-b', '(^_^)', '(^-^)', '(^=^)', '(^.^)', '(~_^)', '(^_~)', '~.^', '^.~', '(^o^)', '(^3^)', '^///^', '>///<', '>///>', 'o///o', '-///-', '=///=', '(V)!_!(V)', '&_&', '0-0', '(\\_/)', 'B)', ':3', 'O|-|_', 'orz', '/\\_/\\', '/\\', '_/\\_', '_/-\\_', '/\\/\\']
Nlist = ['8===D', '8===B', '--!--', ':-(', ':(']

Pdic = {':-)': 'HAPPYFACE', ':)': 'HAPPYFACE', ':o)': 'HAPPYFACE', ':]': 'HAPPYFACE', ':3': 'HAPPYFACE', ':c)': 'HAPPYFACE', ':>': 'HAPPYFACE', '=]': 'HAPPYFACE', '8)': 'HAPPYFACE', '=)': 'HAPPYFACE', ':}': 'HAPPYFACE', ':^)': 'HAPPYFACE', ';-)': 'HAPPYFACE', ';)': 'HAPPYFACE', '*)': 'HAPPYFACE', ';]': 'HAPPYFACE', ';D': 'HAPPYFACE', ':-P': 'HAPPYFACE', ':P': 'HAPPYFACE', 'XP': 'HAPPYFACE', ':-p': 'HAPPYFACE', ':p': 'HAPPYFACE', '=p': 'HAPPYFACE', 'xP': 'HAPPYFACE', ':-b': 'HAPPYFACE', ':b': 'HAPPYFACE', ':-O': 'HAPPYFACE', ':O': 'HAPPYFACE', 'O_O': 'HAPPYFACE', 'o_o': 'HAPPYFACE', 'OwO': 'HAPPYFACE', 'O-O': 'HAPPYFACE', '0_o': 'HAPPYFACE', 'O_o': 'HAPPYFACE', 'O3O': 'HAPPYFACE', 'o0o': 'HAPPYFACE', ';o_o;': 'HAPPYFACE', 'o...o': 'HAPPYFACE', '0w0': 'HAPPYFACE', 'd:-)': 'HAPPYFACE', 'qB-)': 'HAPPYFACE', ':)~': 'HAPPYFACE', ':-)>....': 'HAPPYFACE', ':-X': 'HAPPYFACE', ':X': 'HAPPYFACE', ':-#': 'HAPPYFACE', ':#': 'HAPPYFACE', ':-x': 'HAPPYFACE', ':x': 'HAPPYFACE', ':-*': 'HAPPYFACE', ':*': 'HAPPYFACE', '>:)': 'HAPPYFACE', '>;)': 'HAPPYFACE', '>:-)': 'HAPPYFACE', 'B)': 'HAPPYFACE', 'B-)': 'HAPPYFACE', '8-)': 'HAPPYFACE', '<3': 'HAPPYFACE', '<333': 'HAPPYFACE', '=^_^=': 'HAPPYFACE', '=>.>=': 'HAPPYFACE', '=<_<=': 'HAPPYFACE', '=>.<=': 'HAPPYFACE', '\\o': 'HAPPYFACE', 'o/': 'HAPPYFACE', "d'-'": 'HAPPYFACE', "d'_'": 'HAPPYFACE', "d'-'b": 'HAPPYFACE', "b'_'b": 'HAPPYFACE', 'o/\\o': 'HAPPYFACE', ':u': 'HAPPYFACE', "@}-;-'---": 'HAPPYFACE', '3:00': 'HAPPYFACE', '=]:-)=': 'HAPPYFACE', 'd^_^b': 'HAPPYFACE', 'd-_-b': 'HAPPYFACE', '(^_^)': 'HAPPYFACE', '(^-^)': 'HAPPYFACE', '(^=^)': 'HAPPYFACE', '(^.^)': 'HAPPYFACE', '(~_^)': 'HAPPYFACE', '(^_~)': 'HAPPYFACE', '~.^': 'HAPPYFACE', '^.~': 'HAPPYFACE', '(^o^)': 'HAPPYFACE', '(^3^)': 'HAPPYFACE', '^///^': 'HAPPYFACE', '>///<': 'HAPPYFACE', '>///>': 'HAPPYFACE', 'o///o': 'HAPPYFACE', '-///-': 'HAPPYFACE', '=///=': 'HAPPYFACE', '(V)!_!(V)': 'HAPPYFACE', '&_&': 'HAPPYFACE', '0-0': 'HAPPYFACE', '(\\_/)': 'HAPPYFACE', 'O|-|_': 'HAPPYFACE', 'orz': 'HAPPYFACE', '/\\_/\\': 'HAPPYFACE', '/\\': 'HAPPYFACE', '_/\\_': 'HAPPYFACE', '_/-\\_': 'HAPPYFACE', '/\\/\\': 'HAPPYFACE'}
Ndic = {'8===D': 'SADFACE', '8===B': 'SADFACE', '--!--': 'SADFACE', ':-(': 'SADFACE', ':(': 'SADFACE'}
face = {':-)': 'HAPPYFACE', ':)': 'HAPPYFACE', ':o)': 'HAPPYFACE', ':]': 'HAPPYFACE', ':3': 'HAPPYFACE', ':c)': 'HAPPYFACE', ':>': 'HAPPYFACE', '=]': 'HAPPYFACE', '8)': 'HAPPYFACE', '=)': 'HAPPYFACE', ':}': 'HAPPYFACE', ':^)': 'HAPPYFACE', ';-)': 'HAPPYFACE', ';)': 'HAPPYFACE', '*)': 'HAPPYFACE', ';]': 'HAPPYFACE', ';D': 'HAPPYFACE', ':-P': 'HAPPYFACE', ':P': 'HAPPYFACE', 'XP': 'HAPPYFACE', ':-p': 'HAPPYFACE', ':p': 'HAPPYFACE', '=p': 'HAPPYFACE', 'xP': 'HAPPYFACE', ':-b': 'HAPPYFACE', ':b': 'HAPPYFACE', ':-O': 'HAPPYFACE', ':O': 'HAPPYFACE', 'O_O': 'HAPPYFACE', 'o_o': 'HAPPYFACE', 'OwO': 'HAPPYFACE', 'O-O': 'HAPPYFACE', '0_o': 'HAPPYFACE', 'O_o': 'HAPPYFACE', 'O3O': 'HAPPYFACE', 'o0o': 'HAPPYFACE', ';o_o;': 'HAPPYFACE', 'o...o': 'HAPPYFACE', '0w0': 'HAPPYFACE', 'd:-)': 'HAPPYFACE', 'qB-)': 'HAPPYFACE', ':)~': 'HAPPYFACE', ':-)>....': 'HAPPYFACE', ':-X': 'HAPPYFACE', ':X': 'HAPPYFACE', ':-#': 'HAPPYFACE', ':#': 'HAPPYFACE', ':-x': 'HAPPYFACE', ':x': 'HAPPYFACE', ':-*': 'HAPPYFACE', ':*': 'HAPPYFACE', '>:)': 'HAPPYFACE', '>;)': 'HAPPYFACE', '>:-)': 'HAPPYFACE', 'B)': 'HAPPYFACE', 'B-)': 'HAPPYFACE', '8-)': 'HAPPYFACE', '<3': 'HAPPYFACE', '<333': 'HAPPYFACE', '=^_^=': 'HAPPYFACE', '=>.>=': 'HAPPYFACE', '=<_<=': 'HAPPYFACE', '=>.<=': 'HAPPYFACE', '\\o': 'HAPPYFACE', 'o/': 'HAPPYFACE', "d'-'": 'HAPPYFACE', "d'_'": 'HAPPYFACE', "d'-'b": 'HAPPYFACE', "b'_'b": 'HAPPYFACE', 'o/\\o': 'HAPPYFACE', ':u': 'HAPPYFACE', "@}-;-'---": 'HAPPYFACE', '3:00': 'HAPPYFACE', '=]:-)=': 'HAPPYFACE', 'd^_^b': 'HAPPYFACE', 'd-_-b': 'HAPPYFACE', '(^_^)': 'HAPPYFACE', '(^-^)': 'HAPPYFACE', '(^=^)': 'HAPPYFACE', '(^.^)': 'HAPPYFACE', '(~_^)': 'HAPPYFACE', '(^_~)': 'HAPPYFACE', '~.^': 'HAPPYFACE', '^.~': 'HAPPYFACE', '(^o^)': 'HAPPYFACE', '(^3^)': 'HAPPYFACE', '^///^': 'HAPPYFACE', '>///<': 'HAPPYFACE', '>///>': 'HAPPYFACE', 'o///o': 'HAPPYFACE', '-///-': 'HAPPYFACE', '=///=': 'HAPPYFACE', '(V)!_!(V)': 'HAPPYFACE', '&_&': 'HAPPYFACE', '0-0': 'HAPPYFACE', '(\\_/)': 'HAPPYFACE', 'O|-|_': 'HAPPYFACE', 'orz': 'HAPPYFACE', '/\\_/\\': 'HAPPYFACE', '/\\': 'HAPPYFACE', '_/\\_': 'HAPPYFACE', '_/-\\_': 'HAPPYFACE', '/\\/\\': 'HAPPYFACE', '8===D': 'SADFACE', '8===B': 'SADFACE', '--!--': 'SADFACE', ':-(': 'SADFACE', ':(': 'SADFACE', ':c': 'SADFACE', ':<': 'SADFACE', ':[': 'SADFACE', ':{': 'SADFACE', ':-9': 'SADFACE', 'c.c': 'SADFACE', 'C.C': 'SADFACE', ':-/': 'SADFACE', ':/': 'SADFACE', ':\\': 'SADFACE', '=/': 'SADFACE', '=\\': 'SADFACE', ':S': 'SADFACE', ':|': 'SADFACE', 'O:-)': 'SADFACE', '0:3': 'SADFACE', 'O:)': 'SADFACE', ":'(": 'SADFACE', ';*(': 'SADFACE', 'T_T': 'SADFACE', 'TT_TT': 'SADFACE', 'T.T': 'SADFACE', 'Q.Q': 'SADFACE', 'Q_Q': 'SADFACE', ';_;': 'SADFACE', '^o)': 'SADFACE', '^>.>^': 'SADFACE', '^<.<^': 'SADFACE', '^>_>^': 'SADFACE', '^<_<^': 'SADFACE', 'D:<': 'SADFACE', '>:(': 'SADFACE', 'D-:>': 'SADFACE', '>:-(': 'SADFACE', ':-@[1]': 'SADFACE', ';(': 'SADFACE', "`_'": 'SADFACE', 'D<': 'SADFACE', '(>.<)': 'SADFACE', '(>_>)': 'SADFACE', '(<_<)': 'SADFACE', '(-_-)': 'SADFACE', "(^_^')": 'SADFACE', "^_^_^')": 'SADFACE', '^^"': 'SADFACE', "^^^_.^')": 'SADFACE', '^^_^^;': 'SADFACE', '^&^^.^;&': 'SADFACE', '^^^;': 'SADFACE', '^^^7': 'SADFACE', '9(x.x)9': 'SADFACE', '(;.;)9': 'SADFACE', '(._.)': 'SADFACE', '(,.,)': 'SADFACE', '(X_X)': 'SADFACE', 'x_x': 'SADFACE', '_|_': 'SADFACE', '(-.-)': 'SADFACE', 't(>.<t)': 'SADFACE', 'O?O': 'SADFACE', '(^^^)': 'SADFACE', 'x_O': 'SADFACE', 'O_x': 'SADFACE'}
face1 = {':)': 'HAPPYFACE',':(': 'SADFACE'}

#replace user mentions with "USERMENTION"
def replaceUserMentions(tweet):
	return re.sub("(@[A-Za-z0-9_]+)", "USERMENTION", tweet)
#
# def replaceface(tweet):
#     tweet = re.sub(":\(|:-\(", "HAPPYFACE", tweet)
#     result = re.sub(":\)|:-\\)", "SADFACE", tweet)
#     # result = re.sub("8===D|8===B|\-\-!\-\-|:\-\(|:\(|:c|:<|:\[|:\{|:\-9|c\.c|C\.C|:\-/|:/|:\\|=/|=\\|:S|:\||O:\-\)|0:3|O:\)|:'\(|;\*\(|T_T|TT_TT|T\.T|Q\.Q|Q_Q|;_;|\^o\)|\^>\.>\^|\^<\.<\^|\^>_>\^|\^<_<\^|D:<|>:\(|D\-:>|>:\-\(|:\-@\[1\]|;\(|`_'|D<|\(>\.<\)|\(>\.<\)|\(>_>\)|\(<_<\)|\(\-_\-\)|\(\^_\^'\)|\^_\^_\^'\)|\^\^"|\^\^\^_\.\^'\)|\^\^_\^\^;|\^\&\^\^\.\^;\&|\^\^\^;|\^\^\^7|9\(x\.x\)9|\(;\.;\)9|\(\._\.\)|\(,\.,\)|\(X_X\)|x_x|_\|_|\(\-\.\-\)|_\|_|t\(>\.<t\)|O\?O|\(\^\^\^\)|x_O|O_x", "SADFACE", tweet)
# 	# ':\\)'
#     return result

# def replaceface(tweet):
# 	tweet = re.sub(":\-\)\|:\)\|:o\)\|:\]\|:3\|:c\)\|:>\|=\]\|8\)\|=\)\|:\}\|:\^\)\|:D\|C:\|:\-D\|:D\|8D\|xD\|XD\|=D\|=3\|<=3\|<=8\|;\-\)\|;\)\|\*\)\|;\]\|;D\|:\-P\|:P\|XP\|:\-p\|:p\|=p\|xP\|:\-b\|:b\|:\-O\|:O\|O_O\|o_o\|OwO\|O\-O\|0_o\|O_o\|O3O\|o0o\|;o_o;\|o\.\.\.o\|0w0\|d:\-\)\|qB\-\)\|:\)\~\|:\-\)>\|:\-X\|:X\|:\-\#\|:\#\|:\-x\|:x\|:\-\*\|:\*\|>:\)\|>;\)\|>:\-\)\|B\)\|B\-\)\|8\)\|8\-\)\|<3\|<333\|=\^_\^=\|=>\.>=\|=<_<=\|=>\.<=\|\\,,/\|\\m/\|\\m/\\>\.</\\m/\|\\o/\|\\oo/\|d'\-'\|d'_'\|d'\-'b\|b'_'b\|o/\\o\|:u\|3:00\|=\]:\-\)=\|d\^_\^b\|d\-_\-b\|\(\^_\^\)\|\(\^\-\^\)\|\(\^=\^\)\|\(\^\.\^\)\|\(\~_\^\)\|\(\^_\~\)\|\~\.\^\|\^\.\~\|\(\^o\^\)\|\(\^3\^\)\|d\(>w<\)b\|\^///\^\|>///<\|>///>\|o///o\|\-///\-\|=///=\|\(V\)!_!\(V\)\|0v0\|\|\&_\&\|0\-0\|\(\\_/\)\|B\)\|X3\|:3\|O\|\-\|_\|orz\|/\\_/\\\|/\\\|_/\\_\|_/\-\\_", "SADFACE", tweet)
	# result = re.sub("\-\-!\|\-\-\|:\-\(\|:\(\|:c\|:<\|:\[\|:\{\|D:\|D8\|D;\|D=\|DX\|v\.v\|Dx\|:\-9\|c\.c\|C\.C\|:\-/\|:/\|:\\\|=/\|=\\\|:S\|:\|\|O:\-\)\|0:3\|O:\)\|:'\(\|;\*\(\|T_T\|TT_TT\|T\.T\|Q\.Q\|Q_Q\|;_;\|\^o\)\|\^>\.>\^\|\^<\.<\^\|\^>_>\^\|\^<_<\^\|D:<\|>:\(\|D\-:>\|>:\-\(\|:\-@\[1\]\|;\(\|`_'\|D<\|:\&\|\(>\.<\)\|\(>\.<\)\|\(>_>\)\|\(<_<\)\|\(\-_\-\)\|\(\^_\^'\)\|9\(x\.x\)9\|\(;\.;\)9\|\(\._\.\)\|\(,\.,\)\|\(\-_\-\)\|Zzz\|\(X_X\)\|x_x\|_\|_\|\(\-\.\-\)\|_\|_\|\(t>\.<t\)\|O\?O\|\(\^\^\^\)\|x_O\|O_x", "SADFACE", tweet)
	# return tweet
# def replaceface(tweet):
# 	tweet = re.sub(":\-\)|:\)|:o\)|:\]|:3|:c\)|:>|=\]|8\)|=\)|:\}|:\^\)|;\-\)|;\)|\*\)|;\]|;D|:\-P|:P|XP|:\-p|:p|=p|xP|:\-b|:b|:\-O|:O|O_O|o_o|OwO|O\-O|0_o|O_o|O3O|o0o|;o_o;|o\.\.\.o|0w0|d:\-\)|qB\-\)|:\)\~|:\-\)>\.\.\.\.|:\-X|:X|:\-\#|:\#|:\-x|:x|:\-\*|:\*|>:\)|>;\)|>:\-\)|B\)|B\-\)|8\)|8\-\)|<3|<333|=\^_\^=|=>\.>=|=<_<=|=>\.<=|\\o|o/|d'\-'|d'_'|d'\-'b|b'_'b|o/\\o|:u|@\}\-;\-'\-\-\-|3:00|=\]:\-\)=|d\^_\^b|d\-_\-b|\(\^_\^\)|\(\^\-\^\)|\(\^=\^\)|\(\^\.\^\)|\(\~_\^\)|\(\^_\~\)|\~\.\^|\^\.\~|\(\^o\^\)|\(\^3\^\)|\^///\^|>///<|>///>|o///o|\-///\-|=///=|\(V\)!_!\(V\)|\&_\&|0\-0|\(\\_/\)|B\)|:3|O\|\-\|_|orz|/\\_/\\|/\\|_/\\_|_/\-\\_|/\\/\\", "HAPPYFACE", tweet)
# 	result = re.sub("\-\-!|\-\-|:\-\(|:\(|:c|:<|:\[|:\{|D:|D8|D;|D=|DX|v\.v|Dx|:\-9|c\.c|C\.C|:\-/|:/|:\\|=/|=\\|:S|:\||O:\-\)|0:3|O:\)|:'\(|;\*\(|T_T|TT_TT|T\.T|Q\.Q|Q_Q|;_;|\^o\)|\^>\.>\^|\^<\.<\^|\^>_>\^|\^<_<\^|D:<|>:\(|D\-:>|>:\-\(|:\-@\[1\]|;\(|`_'|D<|:\&|\(>\.<\)|\(>\.<\)|\(>_>\)|\(<_<\)|\(\-_\-\)|\(\^_\^'\)|9\(x\.x\)9|\(;\.;\)9|\(\._\.\)|\(,\.,\)|\(\-_\-\)|Zzz|\(X_X\)|x_x|_\|_|\(\-\.\-\)|_\|_|\(t>\.<t\)|O\?O|\(\^\^\^\)|x_O|O_x", "SADFACE", tweet)
# 	return result
# def replaceface(tweet):
# 	tweet = re.sub(r":\]|:3|:c\)|:>|=\]|8\)|=\)|:\}|:\^\)|:D|C:|:\-D|:D|8D|xD|XD|=D|=3|<=3|<=8", "HAPPYFACE", tweet)
# 	result = re.sub(r"\-\-|:\-\(|:\(|:c|:<|:\[", "SADFACE", tweet)
# 	return result

# def replaceface(tweet):
# 	tweet = re.sub(r"", "HAPPYFACE", tweet)
# 	result = re.sub(r"", "SADFACE", tweet)
# 	return result

def replaceface(tweet):
	tweet = re.sub(r"(\^\^)|(\^\.\^)|(\^\-\^)|(:\-\))|(:\))|(:o\))|(:\])|(:3)|(:c\))|(:>)|(=\])|(8\))|(=\))|(:\})|(:\^\))|(:D)|(C:)|(:\-D)|(:D)|(8D)|(xD)|(XD)|(=D)|(=3)|(<=3)|(<=8)|(;\-\))|(;\))|(\*\))|(;\])|(;D)|(:\-P)|(:P)|(XP)|(:\-p)|(:p)|(=p)|(xP)|(:\-b)|(:b)|(:\-O)|(:O)|(O_O)|(o_o)|(OwO)|(O\-O)|(0_o)|(O_o)|(O3O)|(o0o)|(;o_o;)|(o\.\.\.o)|(0w0)|(d:\-\))|(qB\-\))|(:\)\~)|(:\-\)>)|(:\-X)|(:X)|(:\-\#)|(:\#)|(:\-x)|(:x)|(:\-\*)|(:\*)|(>:\))|(>;\))|(>:\-\))|(B\))|(B\-\))|(8\))|(8\-\))|(<3)|(<333)|(=\^_\^=)|(=>\.>=)|(=<_<=)|(=>\.<=)|(\\,,/)|(\\m/)|(\\m/\\>\.</\\m/)|(\\o/)|(\\oo/)|(d'\-')|(d'_')|(d'\-'b)|(b'_'b)|(o/\\o)|(:u)|(3:00)|(=\]:\-\)=)|(d\^_\^b)|(d\-_\-b)|(\(\^_\^\))|(\(\^\-\^\))|(\(\^=\^\))|(\(\^\.\^\))|(\(\~_\^\))|(\(\^_\~\))|(\~\.\^)|(\^\.\~)|(\(\^o\^\))|(\(\^3\^\))|(d\(>w<\)b)|(\^///\^)|(>///<)|(>///>)|(o///o)|(\-///\-)|(=///=)|(\(V\)!_!\(V\))|(0v0)|(\\m/\*\.\*\\m/)|(\&_\&)|(0\-0)|(\(\\_/\))|(B\))|(X3)|(:3)|(O\|\-\|_)|(orz)|(/\\_/\\)|(/\\)|(_/\\_)|(_/\-\\_)|(/\\/\\)", "HAPPYFACE", tweet)
	result = re.sub(r"(\-\-!)|(\-\-)|(:\-\()|(:\()|(:c)|(:<)|(:\[)|(:\{)|(D:)|(D8)|(D;)|(D=)|(DX)|(v\.v)|(Dx)|(:\-9)|(c\.c)|(C\.C)|(:\-/)|(:/)|(:\\)|(=/)|(=\\)|(:S)|(:\|)|(O:\-\))|(0:3)|(O:\))|(:'\()|(;\*\()|(T_T)|(TT_TT)|(T\.T)|(Q\.Q)|(Q_Q)|(;_;)|(\^o\))|(\^>\.>\^)|(\^<\.<\^)|(\^>_>\^)|(\^<_<\^)|(D:<)|(>:\()|(D\-:>)|(>:\-\()|(:\-@\[1\])|(;\()|(`_')|(D<)|(:\&)|(\(>\.<\))|(\(>\.<\))|(\(>_>\))|(\(<_<\))|(\(\-_\-\))|(\(\^_\^'\))|(9\(x\.x\)9)|(\(;\.;\)9)|(\(\._\.\))|(\(,\.,\))|(\(\-_\-\))|(Zzz)|(\(X_X\))|(x_x)|(_\|_)|(\(\-\.\-\))|(_\|_)|(\(t>\.<t\))|(O\?O)|(\(\^\^\^\))|(x_O)|(O_x)", "SADFACE", tweet)
	return result
# def replaceface(tweet):
#     # tweet = re.sub(r":(|:-(", "HAPPYFACE", tweet)
#     # result = re.sub(":)|:-)", "SADFACE", tweet)
# 	# pattern = re.compile(r'\b(' + '|'.join(face.keys()) + r')\b')
# 	# result = pattern.sub(lambda x: face[x.group()], tweet)
# 	# print('0000000000')
# 	# for key in face:
# 	# 	print(key)
# 	# 	# print(face[key])
# 	# 	result = tweet.replace(key, face[key])
# 	# 	result = result.replace('asdg', 'hahh')
# 	# 	result = result.replace(':)', 'myhaapp')
# 	for key in face1:
# 		result = tweet.replace(key, face[key])
# 	return result

def replaceNum_1letter(tweet):
	tweet = ' '.join(pattern_no_num.findall(tweet))
	result = ' '.join(pattern_no_1letter.findall(tweet))
	return result
#replace all non-alphanumeric
def replaceRest(tweet):
	result = re.sub("[^a-zA-Z0-9]", " ", tweet)
	return re.sub(' +',' ', result)

def replaceLong(tweet):
	return re.sub(r'([a-zA-Z])(\1){2,}', r'\1', tweet)

def replacelol(tweet):
	return re.sub(r'\b(lol)\b', 'HAPPYFACE', tweet)

def lower_stem(tweet):
	result = tweet.lower()
	sno = nltk.stem.SnowballStemmer('english')


def replaceall(tweet):
	# num = 0
	tweet = replaceface(tweet)
	tweet = replaceURLs(tweet)
	tweet = replaceUserMentions(tweet)
	tweet = replaceLong(tweet)
	tweet = replacelol(tweet)
	tweet = replaceNum_1letter(tweet)
	# tweet1 = replaceRest(tweet)
	# print(tweet)
	# if(tweet1 != tweet):
	# 	# num += 1
	# 	print('something wrong?----------------------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	# print(tweet1)
	return tweet

def testit():
	with open("mytweets.json", 'r') as f:
		for line in f:
			text = json.loads(line)['text']
			newtext = nltk_tokenise(text).lower()
			newtext = replaceURLs(newtext)
			newtext1 = replaceUserMentions(newtext)
			newtext = replaceRest(newtext1)
			if(newtext != newtext1):
				print('something wrong?')
			print(text + '\n' + newtext + '\n')
	f.close()

# s = 'asdg  :) :(  :-) O_x  :-9  ^.^'
# ans = replaceall(s)
# print(ans)
# s = 'asdg lol dflol :)'
# ans = replaceall(s)
# print(ans)
# print(ans)
#
# s = 'Спорт not russianA'
# d = {
# 'Спорт':'Досуг',
# 'russianA':'englishA'
# }
#
# pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')
# result = pattern.sub(lambda x: d[x.group()], s)
# print(result)