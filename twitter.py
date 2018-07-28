# Twitter API Blockchain Legal Issues & Ethics Investigation
# You Jeen Ha
# 27 July 2018

import json, tweepy, re, operator, nltk, string, sys, vincent, random
nltk.download('stopwords')
import pandas as pd 
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from metakernel.display import display

consumer_key = '1fPYEdUpolPOl9hgJufgkbhHu'
consumer_secret = 'rAKtDHVcKnpMGNSF7XJLdf0nmDsJQXiZcPeFUkZYbvSnrgJMEa'
access_token = '974496587698327552-M6e6wAWA0GIUO48Tp8EKu2oVV527miK'
access_secret= 'QjdUPmBtkybwXAEl9P6Kp2RM2Zty4DQCL40bhNisHsS9n'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

''' Process or store tweet '''
def process_or_store(tweet):
	json.dumps(tweet)


for status in tweepy.Cursor(api.home_timeline).items(10):
	# Process a single status
	process_or_store(status._json)


'''Keeps connection open to streaming'''
# class MyListener(StreamListener):

# 	def on_data(self, data):
# 		try: 
# 			with open('python.json', 'a') as f:
# 				f.write(data)
# 				return True
# 		except BaseException as e:
# 			print("Error on_data: %s" % str(e))
# 		return True

# 	def on_error(self, status):
# 		print(status)
# 		return True


# twitter_stream = Stream(auth, MyListener())
# twitter_stream.filter(track= ['#blockchain'])


############################ TOKENIZE ######################################
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

''' Tokenize the tweet. '''
def tokenize(s):
    return tokens_re.findall(s)
 
''' Preprocess tokens such as emoticons, URLs, and hashtags. '''
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

# tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
# print(preprocess(tweet))

# STOP WORDS
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

# TERM REOCCURENCES
com = defaultdict(lambda : defaultdict(int))

###################### ANALYZE THE JSON DATASET ##############################
filename = 'python.json'
with open(filename, 'r') as f:
	
	search_word = sys.argv[1] # pass a term as a command-line argument
	
	dates_bitcoin = []
	# dates_cryptocurrency = []
	# dates_law = []

	count_search = Counter()
	
	for line in f:
		
		tweet = json.loads(line) # load it as Python dict
		keys = tweet.keys()
		
		##############################################################
		if 'text' in tweet:
			# terms_all = [term for term in preprocess(tweet['text'])]
			# Stop word removal
			terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]
			
			# CUSTOMIZE LIST OF INTERESTING TERMS/TOKENS
			# terms_single = set(terms_all) # count terms only once
			terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#')]
			# count terms only (no hashtags, no mentions)
			terms_only = [term for term in preprocess(tweet['text']) if term not in stop and not term.startswith(('#', '@'))]
			# mind the ((double brackets))
			# startswith() takes a tuple, not a list if we pass a list of inputs

			# terms_bigram = bigrams(terms_stop)

			#Update counter
			# if search_word in terms_only:
			# 	count_search.update(terms_only)

			if search_word in terms_hash:
				count_search.update(terms_hash)

		
			############# TIME SERIES VISUALIZATION #########################
			if '#bitcoin' in terms_hash:
				dates_bitcoin.append(tweet['created_at'])
				# dates_cryptocurrency.append(tweet['created_at'])
				# dates_law.append(tweet['created_at'])
	
	# Create a date range and populate with random data
	# idx = pd.DatetimeIndex(dates_bitcoin)

	# GET FREQUENCY
	bitcoin = [count_search for i in range(len(dates_bitcoin))]
	bitcoin_series = pd.Series(multi_iter1, index=dates_bitcoin)


	# Put time series in plot with Vincent
	time_chart = vincent.Line(bitcoin_series)
	time_chart.scales[0].type = 'ordinal'

	# Make visualization wider and add axis titles
	time_chart.axis_titles(x='Time', y='Freq')
	time_chart.legend(title='Bitcoin')
	time_chart.to_json('time_chart.json')
	time_chart.display()

	# # all the data together
	# match_data = dict(bitcoin = per_minute_i, cryptocurrency = per_minute_j, law = per_minute_k)
	# # we need a DataFrame to accommodate multiple series
	# all_matches = pandas.DataFrame(data=match_data, index=per_minute_i.index)
	# # Resampling as above
	# all_matches = all_matches.resample('1Min', how='sum').fillna(0)
	# # Plotting
	# time_char = vincent.Line(all_matches[['bitcoin', 'cryptocurrency', 'law']])
	# time_chart.axis_titles(x='Time', y='Freq')
	# time_chart.legend(title='Matches')
	# time_chart.to_json('time_chart.json')

	################### BUILD CO-OCCURRENCE MATRIX ####################
	for i in range(len(terms_only) - 1):
		for j in range(i+1, len(terms_only)):
			w1, w2 = sorted([terms_only[i], terms_only[j]])
			if w1 != w2:
				com[w1][w2] += 1

	com_max = []
	# For each term, look for most common co-occurrence
	for t1 in com:
		t1_max_terms = sorted(com[t1].items(), key = operator.itemgetter(1), reverse=True)[:5]
		for t2, t2_count in t1_max_terms:
			com_max.append(((t1, t2), t2_count))
	# # Get the most frequent co-occurrences
	# terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
	# # print(terms_max[:5])

	# Print first 5 most frequent words
	# print(count_all.most_common(5))

	#Print Co-occurrences
	# print("Co-occurrence for %s:" % search_word)
	# print(count_search.most_common(20))

	


