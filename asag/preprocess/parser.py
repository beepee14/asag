from os import listdir
from os.path import isfile, join
import xml.etree.cElementTree as ET

def get_file_names(dir_path):
	return [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]

def parse_file(file_path):
	tree = ET.ElementTree(file=file_path)
	root = tree.getroot()
	question = root[0].text
	reference_answers = []
	for child in root[1]:
		reference_answers.append((child.text,child.attrib["category"]))
	student_answers = []
	for child in root[2]:
		student_answers.append((child.text,child.attrib["accuracy"]))
	return [question,reference_answers,student_answers]

def get_data(dir_path):
	file_names = get_file_names(dir_path)
	data = []
	for file_name in file_names:
		file_path = dir_path + "/" + file_name
		data.append(parse_file(file_path))
	return data

def main():
	dir_path = "../data/semeval2013-Task7-2and3way/training/2way/beetle"
	data = get_data(dir_path)

if __name__ == '__main__':
	main()