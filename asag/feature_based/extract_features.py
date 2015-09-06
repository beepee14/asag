""" feature Extraction for building the feature set """

"""Implementation of various similarity measures """

from nltk.align.bleu_score import bleu
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

EN_STOPWORDS = stopwords.words('english')
brown_ic = wordnet_ic.ic('ic-brown.dat')

def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wn.ADJ
	elif treebank_tag.startswith('V'):
		return wn.VERB
	elif treebank_tag.startswith('N'):
		return wn.NOUN
	elif treebank_tag.startswith('R'):
		return wn.ADV
	else:
		return ''

def get_word_similarity(word,other_word,metric):
	word_synsets = wn.synsets(word[0], pos = get_wordnet_pos(word[1]))
	if len(word_synsets) == 0 :
		word_synsets = wn.synsets(word[0])
	other_word_synsets = wn.synsets(other_word[0], pos =get_wordnet_pos(other_word[1]))
	if len(other_word_synsets) == 0:
		other_word_synsets = wn.synsets(other_word[0])
	maxi = 0
	for syn in word_synsets:
		for other_syn in other_word_synsets:
			if other_syn.pos == syn.pos:
				try:
					if metric == "wup":
						maxi = max(maxi, min(10, syn.wup_similarity(other_syn)))
					if metric == "lch":
						maxi = max(maxi, min(10, syn.lch_similarity(other_syn)))
					if metric == "res":
						maxi = max(maxi, min(10, syn.res_similarity(other_syn, brown_ic)))
					if metric == "lin":
						maxi = max(maxi, min(10, syn.lin_similarity(other_syn, brown_ic)))
				except :
					maxi = maxi
	return maxi

def max_sim(word, sentence, metric):
	if len(sentence) == 0:
		return 0
	max_sim = get_word_similarity(word,sentence[0],metric)
	for i in range(1,len(sentence)):
		max_sim = max(max_sim, get_word_similarity(word,sentence[i],metric))
	return max_sim

def get_text_similarity(string1_tokens, string2_tokens, metric):
	sim_score = 0.0
	sum_score = 0.0
	if len(string1_tokens) != 0:
		for word in string1_tokens:
			sum_score += max_sim(word,string2_tokens,metric)
		sim_score += float(sum_score)/len(string1_tokens)
	sum_score = 0.0
	if len(string2_tokens) != 0:
		for word in string2_tokens:
			sum_score += max_sim(word,string1_tokens,metric)
		sim_score += float(sum_score)/len(string2_tokens)
	sim_score = sim_score/2
	return sim_score

def get_bleu_similarity(reference_answers, student_answer):
	porter_stemmer = PorterStemmer()
	reference_answers_tokens = []
	for answer in reference_answers:
		reference_answers_tokens.append(map(lambda x: str(porter_stemmer.stem(x)), answer.split()))
	student_answer = map(lambda x: str(porter_stemmer.stem(x)), student_answer.split())
	weights = [0.25, 0.25]
	return bleu(student_answer,reference_answers_tokens, weights)

def main():
	string1 = "this is a very big house"
	string2 = "the house was quite large"
	print get_text_similarity(string1, string2, "wup")

if __name__ == '__main__':
	main()
