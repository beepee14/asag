# Preprocess the student answers
import parser
import spell
import re,string

def remove_punctuation(_text):
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	text = regex.sub('', _text)
	return text.lower()

def get_all_words(data):
	all_words = set()
	for data_point in data:
		question = remove_punctuation(data_point[0])
		for word in question.split():
			all_words.add(word)
		for reference_answer in  data_point[1]:
			answer = remove_punctuation(reference_answer[0])
			for word in answer.split():
				all_words.add(word)
	return list(all_words)

def correct_answer(NWORDS, all_words,student_answers):
	spell_corrected_answers = []
	for answer in student_answers:
		answer_words = []
		for word in answer[0].split():
			answer_words.append(spell.correct(word, NWORDS))
		answer_string = " ".join(answer_words)
		spell_corrected_answers.append((answer_string,answer[1]))
	return spell_corrected_answers

def correct_student_answers(data):
	spell_corrected_data = []
	all_words = get_all_words(data)
	NWORDS = spell.compute_nwords(all_words)
	for data_point in data:
		spell_corrected_answers = correct_answer(NWORDS,all_words,data_point[2])
		spell_corrected_data.append([data_point[0],
									data_point[1],
									spell_corrected_answers])
	return spell_corrected_data
	
def main():
	dir_path = "../data/2-3way_SemEvalData/2way/beetle"
	data = parser.get_data(dir_path)
	data = correct_student_answers(data)

if __name__ == '__main__':
	main()