"""Implementation of various similarity measures """
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

def get_cosine_similarity(string1, string2):
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform([string1,string2])
	X_train_counts = X_train_counts.todense()
	return 1 - spatial.distance.cosine(X_train_counts[0],X_train_counts[1])

def computeLeskSimilarity():
	return

def main():
	dir_path = "../data/semeval2013-Task7-2and3way/training/2way/beetle"
	data = parser.get_data(dir_path)
	data = preprocess.correct_student_answers(data)

if __name__ == '__main__':
	main()