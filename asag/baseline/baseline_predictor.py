""" Baseline predictor """

from preprocess.parser import get_data
from preprocess.preprocess import correct_student_answers
from scipy import spatial
from similarity_measures import get_cosine_similarity
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def get_all_features_question(question, student_answer):
	cosine_sim = get_cosine_similarity(question,student_answer)
	return [cosine_sim]

def get_all_features_ref(reference_answers, student_answer):
	max_cosine_sim = get_cosine_similarity(reference_answers[0],student_answer)
	for i in range(1,len(reference_answers)):
		cosine_sim = get_cosine_similarity(reference_answers[i],student_answer)
		if cosine_sim > max_cosine_sim :
			max_cosine_sim = cosine_sim
	return [max_cosine_sim]

def get_single_data(data_point):
	question = data_point[0]
	reference_answers = [ans for ans, _ in data_point[1]]
	feature_data = []
	for student_answer, label in data_point[2]:
		feature_row = get_all_features_question(question, student_answer)
		feature_row = feature_row + get_all_features_ref(reference_answers, student_answer)
		feature_row = feature_row + [1 if label=="correct" else 0]
		feature_data.append(feature_row)
	return feature_data

def get_feature_data(dir_path):
	data = get_data(dir_path)
	data = correct_student_answers(data)
	feature_data = []
	for data_point in data:
		feature_data += get_single_data(data_point)
	return feature_data

def split_labels(data):
	X = []
	y = []
	for row in data:
		X.append(row[:-1])
		y.append(row[-1])
	return X, y

def compute_accuracy(pred_Y,test_Y):
	correct_count = 0
	for i in range(len(pred_Y)):
		if pred_Y[i] == test_Y[i] :
			correct_count += 1
	return (float(correct_count))/len(pred_Y)

def fit_predict(data_train, data_test):
	train_X, train_Y = split_labels(data_train)
	test_X, test_Y = split_labels(data_test)
	clf = RandomForestClassifier(n_estimators=10)
	clf = clf.fit(train_X, train_Y)
	pred_Y = clf.predict(test_X)
	accuracy = compute_accuracy(pred_Y, test_Y)
	print accuracy

def main():
	dir_path = "../data/semeval2013-Task7-2and3way/training/2way/beetle"
	data_train = get_feature_data(dir_path)
	dir_path = "../data/semeval2013-Task7-2and3way/test/2way/beetle/test-unseen-answers"
	data_test = get_feature_data(dir_path)
	fit_predict(data_train, data_test)

if __name__ == '__main__':
	main()