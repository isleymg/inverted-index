from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import stopwords

import json
import math
import os
import re

# key = term, value = {key = document directory/file id, value = term frequency}
TF_MAP = defaultdict(lambda: defaultdict(float))

# key = term, value = {key = document directory/file id, value = term count}
H1_MAP = defaultdict(lambda: defaultdict(int))
H2_MAP = defaultdict(lambda: defaultdict(int))
H3_MAP = defaultdict(lambda: defaultdict(int))
B_MAP = defaultdict(lambda: defaultdict(int))

# key = term, value = {key = document directory/file id, value = idf score}
IDF_MAP = defaultdict(lambda: defaultdict(float))

# key = term, value = {key = document directory/file id, value = tf-idf score}
TF_IDF_MAP = defaultdict(lambda: defaultdict(float))

# number of documents in the corpus
DOCUMENT_COUNT = 1021

def get_all_file_paths():
	base_file_directory = 'WEBPAGES_SIMPLE'

	for root, dirs, _ in os.walk(base_file_directory):
		for d in dirs:
			sub_dir = root + '/' + d

			for _, _, files in os.walk(sub_dir):
				for f in files:
					file_name = sub_dir + '/' + f

					yield file_name

def get_words(soup_text):
	stop_words = set(stopwords.words('english'))

	# remove absolute urls, file type extensions, numbers
	text = re.sub(r'http.*|^(\/)+|[0-9]+|(\.)+[a-zA-Z]*[\/]*', '', soup_text).encode('UTF-8')

	# replace non-alphabet characters but keep apostrophes by single space
	data = re.sub(r'[^a-zA-Z\']+', ' ', text)

	# replace words in quotes by empty string
	data = re.sub(r'[\']+([a-zA-Z]+)[\']+', '', data)

	# remove all remaining quotes from start/end of string
	data = re.sub(r'\'+\s|\s\'+', ' ', data)

	tokens = map(str.lower, data.split())

	# filters out single-letter or stop words
	words = list(filter(lambda x: (len(x) > 1 and x not in stop_words), tokens))	

	return words		

def build_inverted_index():
	important_tags = ['h1', 'h2', 'h3', 'b']

	# key = document directory/file id, value = total term count
	document_counts = defaultdict(int)

	for path in get_all_file_paths():
		html = open(path, 'r')
		soup = BeautifulSoup(html, 'lxml')
		words = get_words(soup.text)

		# ~/WEBPAGES_SIMPLE/directory_id/file_id
		directory_id, file_id = path.split('/')[-2:]

		for w in words:
			directory_file_id = '{0}/{1}'.format(directory_id, file_id)
			document_counts[directory_file_id] += 1

			# temporarily stores term counts
			# value will be updated to term frequency once document_counts is built
			TF_MAP[w][directory_file_id] += 1	

		for tag in important_tags:
			content = ' '.join(i.text for i in soup.findAll(tag))
			tag_words = get_words(content)
			map_name = '{0}_MAP'.format(tag.upper())
			map_obj = eval(map_name)

			for w in tag_words:
				map_obj[w]['{0}/{1}'.format(directory_id, file_id)] += 1

	for term, values in TF_MAP.items():
		for directory_file_id, term_count in values.items():
			doc_count = document_counts[directory_file_id]

			# prevent division by 0
			if doc_count == 0:
				doc_count = 1

			# update map to store actual tf values
			TF_MAP[term][directory_file_id] /= float(doc_count)

	for i in ['tf'] + important_tags:
		file_name = '{0}.json'.format(i)
		map_name = '{0}_MAP'.format(i.upper())
		map_obj = eval(map_name)

		with open(file_name, 'w') as f:
			json.dump(map_obj, f)

def calculate_idf():
	for term, values in TF_MAP.items():
		# prevent division by 0
		idf = float(DOCUMENT_COUNT) / (len(values) if len(values) > 0 else 1)
		IDF_MAP[term] = idf		

	with open('idf.json', 'w') as f:
		json.dump(IDF_MAP, f)	

def calculate_tf_idf():
	for term, values in TF_MAP.items():
		idf = IDF_MAP[term]

		for directory_file_id, tf in values.items():
			extra_score = H1_MAP.get(term, {}).get(directory_file_id, 0) + \
				H2_MAP.get(term, {}).get(directory_file_id, 0) + \
				H3_MAP.get(term, {}).get(directory_file_id, 0) + \
				B_MAP.get(term, {}).get(directory_file_id, 0)
			tf_idf = math.log(1 + tf) * math.log(idf)

			if extra_score > 0:
				tf_idf += math.log(extra_score)

			TF_IDF_MAP[term][directory_file_id] = tf_idf

	with open('tf_idf.json', 'w') as f:
		json.dump(TF_IDF_MAP, f)

def load_json():
	#types = ['tf', 'h1', 'h2', 'h3', 'b']
	types = ['tf', 'h1', 'h2', 'h3', 'b', 'idf', 'tf_idf']

	for t in types:
		file_name = '{0}.json'.format(t)
		map_name = '{0}_MAP'.format(t.upper())

		with open(file_name, 'r') as f:
			globals()[map_name] = json.load(f)		

def milestone1():
	unique_words_count = len(TF_MAP)			

	with open('inverted_index.txt', 'w') as f:
		f.write("{0:35} {1:10}\n".format("Term", "(Doc ID, Frequency)"))

		for term, values in TF_MAP.items():
			counts = []

			for directory_file_id, tf in values.items():
				counts.append((directory_file_id, tf))
 			
 			f.write("{0:35} {1}\n".format(term, ", ".join(map(str, counts))))

		f.write("Number of documents: {0}\n".format(DOCUMENT_COUNT))
		f.write("Number of unique words: {0}\n".format(unique_words_count))


	with open("tf_idf_scores.txt", "w") as f:
		f.write("{0:35} {1:10}\n".format("Term", "(Doc ID, Tf-Idf Score)"))

		for term, values in TF_IDF_MAP.items():
			scores = []

			for directory_file_id, tf_idf in values.items():
				scores.append((directory_file_id, tf_idf))

			f.write("{0:35} {1}\n".format(term, ", ".join(map(str, scores))))	

def milestone2():
	pass						

if __name__ == '__main__':
	#build_inverted_index()
	#load_json()
	#calculate_idf()
	#calculate_tf_idf()
	#milestone1()
	load_json()