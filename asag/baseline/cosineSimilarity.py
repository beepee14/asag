import parser
import spell
import preprocess
def computeCosineSimilarity():

def main():
	dir_path = "../data/semeval2013-Task7-2and3way/training/2way/beetle"
	data = parser.get_data(dir_path)
	data = preprocess.correct_student_answers(data)

if __name__ == '__main__':
	main()