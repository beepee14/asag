""" Feature Based Machine learning model predictor """

from preprocess.parser import get_data
from preprocess.preprocess import correct_student_answers
from baseline.similarity_measures import get_cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import pickle
import extract_features as ex
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
import numpy as np
import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

EN_STOPWORDS = set(stopwords.words('english'))
metrics = ["wup", "res", "lch", "lin"]

import warnings

def vec_similarity_sentences(sentence1, sentence2):
	if len(sentence1) == 0 or len(sentence2)==0:
		return 0
	input_file = 'test.txt'
	sent_file = 'sent.txt'
	
	f = open(sent_file,'w')
	f.write(sentence1)
	f.close()
	model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
	model.save_sent2vec_format(sent_file + '.vec')
	lines = [line.rstrip('\n') for line in open(sent_file + '.vec')][1:]
	lines = lines[0].split()[1:]
	sentence1_rep = [float(i) for i in lines]

	f = open(sent_file,'w')
	f.write(sentence2)
	f.close()
	model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
	model.save_sent2vec_format(sent_file + '.vec')
	
	lines = [line.rstrip('\n') for line in open(sent_file + '.vec')][1:]
	lines = lines[0].split()[1:]
	sentence2_rep = [float(i) for i in lines]
	return 1 - spatial.distance.cosine(sentence1_rep, sentence2_rep)

def vec_similarity_ref_answers(reference_answers, student_answer):
	maxi = -1000000
	for ans in reference_answers:
		simi = vec_similarity_sentences(ans, student_answer)
		if maxi<(simi):
			maxi = simi
	return maxi

def get_all_features_question(question, student_answer, student_answer_tokens):
	cosine_sim = get_cosine_similarity(question,student_answer)
	bleu_sim = ex.get_bleu_similarity([question], student_answer)
	features = [cosine_sim, bleu_sim]

	question_tokens = word_tokenize(question)
	question_tokens = nltk.pos_tag(question_tokens)
	question_tokens = [x for x in question_tokens if x[0] not in EN_STOPWORDS]
	for metric in metrics:
		sim = ex.get_text_similarity(question_tokens,student_answer_tokens,metric)
		features.append(sim)
	features.append(vec_similarity_sentences(question, student_answer))
	return features

def get_all_features_ref(reference_answers, student_answer, student_answer_tokens):
	""" Bleu Similarity"""
	bleu_sim = ex.get_bleu_similarity(reference_answers, student_answer)
	""" Cosine distance """
	max_cosine_sim = get_cosine_similarity(reference_answers[0],student_answer)
	for i in range(1,len(reference_answers)):
		cosine_sim = get_cosine_similarity(reference_answers[i],student_answer)
		if cosine_sim > max_cosine_sim :
			max_cosine_sim = cosine_sim
	features = [max_cosine_sim, bleu_sim]
	""" Wu Palmer and other similarity measures used for text Similarity """
	reference_answers_tokens = []
	for answer in reference_answers:
		reference_answer_tokens = word_tokenize(answer)
		reference_answer_tokens = nltk.pos_tag(reference_answer_tokens)
		reference_answer_tokens = [x for x in reference_answer_tokens if x[0] not in EN_STOPWORDS]
		reference_answers_tokens.append(reference_answer_tokens)
	for metric in metrics:
		max_sim = ex.get_text_similarity(reference_answers_tokens[0],student_answer_tokens, metric)
		for i in range(1,len(reference_answers)):
			sim = ex.get_text_similarity(reference_answers_tokens[i],student_answer_tokens, metric)
			if sim > max_sim :
				max_sim = sim
		features.append(max_sim)
	features.append(vec_similarity_ref_answers(reference_answers, student_answer))
	return features

def get_student_answer_tokens(student_answer):
	student_answer_tokens = word_tokenize(student_answer)
	student_answer_tokens = nltk.pos_tag(student_answer_tokens)
	student_answer_tokens = [x for x in student_answer_tokens if x[0] not in EN_STOPWORDS]
	return student_answer_tokens

def get_single_data(data_point):
	question = data_point[0]
	reference_answers = [ans for ans, _ in data_point[1]]
	feature_data = []
	for student_answer, label in data_point[2]:
		student_answer_tokens = get_student_answer_tokens(student_answer)
		feature_row = get_all_features_question(question, student_answer, student_answer_tokens)
		feature_row = feature_row + get_all_features_ref(reference_answers, student_answer, student_answer_tokens)
		feature_row = feature_row + [1 if label=="correct" else 0]
		feature_data.append(feature_row)
	return feature_data

def compute_accuracy(pred_Y,test_Y):
	correct_count = 0
	for i in range(len(pred_Y)):
		if pred_Y[i] == test_Y[i] :
			correct_count += 1
	return (float(correct_count))/len(pred_Y)

def split_labels(data):
	X = []
	y = []
	for row in data:
		X.append(row[:-1])
		y.append(row[-1])
	return X, y

def count_class(x):
	zeroes = len(np.where(np.array(x) == 0)[0])
	print zeroes, len(x) - zeroes

def fit_predict(data_train, data_test):
	print data_train[:5]
	print data_test[:5]
	train_X, train_Y = split_labels(data_train)
	test_X, test_Y = split_labels(data_test)
	forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
	clf = forest.fit(train_X, train_Y)
	importances = forest.feature_importances_
	pred_Y = clf.predict(test_X)
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
	indices = np.argsort(importances)[::-1]
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(train_X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
	plt.xticks(range(train_X.shape[1]), indices)
	plt.xlim([-1, train_X.shape[1]])
	plt.show()
	count_class(test_Y)
	count_class(pred_Y)

	# accuracy = compute_accuracy(pred_Y, test_Y)
	f1_sco = f1_score(test_Y, pred_Y, average='macro') 
	print f1_sco

def get_all_text(data):
	input_txt = ""
	for data_point in data:
		question = data_point[0]
		input_txt+=question
		input_txt+=". "

		for reference_answer in  data_point[1]:
			answer = reference_answer[0]
			input_txt+=answer
			input_txt+=". "
	return input_txt

def initialise_model(data):
	input_file = 'test.txt'
	f = open(input_file,'w')
	input_txt = get_all_text(data)
	f.write(input_txt)
	f.close()
	model = Word2Vec(LineSentence(input_file), size=100, window=5, sg=0, min_count=1, workers=8)
	model.save(input_file + '.model')
	model.save_word2vec_format(input_file + '.vec')

def get_feature_data(dir_path):
	data = get_data(dir_path)
	data = correct_student_answers(data)
	# initialise_model(data)
	feature_data = []
	counter = 0
	last = 0
	for i in range(len(data)):
		if (i*10)/len(data)>last:
			last = (i*10)/len(data)
			print str(last)+"0% completed"
		feature_data += get_single_data(data[i])
	print "feature extraction complete"
	return feature_data

def main():
	logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.ERROR)
	logging.info("running %s" % " ".join(sys.argv))
	dir_path = "../data/semeval2013-Task7-2and3way/training/2way/beetle"
	data_train = get_feature_data(dir_path)
	print data_train[1]
	dir_path = "../data/semeval2013-Task7-2and3way/test/2way/beetle/test-unseen-answers"
	data_test = get_feature_data(dir_path)
	fit_predict(data_train, data_test)

if __name__ == '__main__':
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		main()