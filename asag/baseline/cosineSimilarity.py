import parser
import spell
import preprocess

def main():
	dir_path = "../data/2-3way_SemEvalData/2way/beetle"
	data = parser.get_data(dir_path)
	data = preprocess.correct_student_answers(data)

if __name__ == '__main__':
	main()