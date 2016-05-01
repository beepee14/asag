from os import listdir
from os.path import isfile, join
import xml.etree.cElementTree as ET
import re,string

def remove_punctuation(_text):
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	text = regex.sub('', _text)
	return text.lower()

# returns a list containing the names of all the files in the directory
def get_file_names(dir_path):
	return [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) and f != ".DS_Store" ]

def parse_file(file_path):
	tree = ET.ElementTree(file=file_path)
	root = tree.getroot()
	question = remove_punctuation(root[0].text)
	reference_answers = []
	for child in root[1]:
		reference_answers.append((remove_punctuation(child.text),child.attrib["category"]))
	student_answers = []
	for child in root[2]:
		student_answers.append((remove_punctuation(child.text),child.attrib["accuracy"]))
	return [question,reference_answers,student_answers]

def get_data(dir_path):
	file_names = get_file_names(dir_path)
	data = []
	print "Parsing of xml files started"
	for i in range(len(file_names)):
		file_path = dir_path + "/" + file_names[i]
		file_data = parse_file(file_path)
		data.append(file_data)
	print "Parsing completed of files from " + dir_path
	return data

def main():
	dir_path = "../data/semeval2013-Task7-2and3way/training/2way/beetle"
	data = get_data(dir_path)

if __name__ == '__main__':
	main()