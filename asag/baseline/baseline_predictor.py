""" Baseline predictor """

from preprocess.parser import get_data
from preprocess.preprocess import correct_student_answers
from scipy import spatial
from similarity_measures import get_cosine_similarity
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

def get_single_train(data_point):
	question = data_point[0]
	reference_answers = [ans for ans, _ in data_point[1]]
	train_data = []
	for student_answer, label in data_point[2]:
		train_row = get_all_features_question(question, student_answer)
		train_row = train_row + get_all_features_ref(reference_answers, student_answer)
		train_row = train_row + [1 if label=="correct" else 0]
		train_data.append(train_row)
	return train_data

def get_training_data(data):
	train_data = []
	for data_point in data:
		train_data += get_single_train(data_point)

def main():
	dir_path = "../data/semeval2013-Task7-2and3way/training/2way/beetle"
	data = get_data(dir_path)
	data = correct_student_answers(data)
	train_data = get_training_data(data)

if __name__ == '__main__':
	main()