"""Implementation of various similarity measures """
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import stopwords

# EN_STOPWORDS = stopwords.words('english')

def get_cosine_similarity(string1, string2):
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform([string1,string2])
	X_train_counts = X_train_counts.todense().tolist()
	if (sum(map(lambda x:x*x,X_train_counts[0])) == 0) or (sum(map(lambda x:x*x,X_train_counts[1])) == 0):
		cosine_sim = 0.0
	else:
		cosine_sim = 1 - spatial.distance.cosine(X_train_counts[0],X_train_counts[1])
	return cosine_sim

# def get_lesk_similarity(string1, string2):


def main():
	string1 = ""
	string2 = ""
	print get_lesk_similarity(string1,string2)

if __name__ == '__main__':
	main()