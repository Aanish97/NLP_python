import re
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import preprocess_string

path_data = "D:/.Semester 7/IR/Assignments/assignment 3/Datasets/Question1.txt"

with open(path_data, "r", encoding='ISO-8859-1') as f:
    f_reader = f.readlines()

preProcessedData = []
i=0

for line in f_reader:
	token =[]
	token = preprocess_string(line)
	preProcessedData.append(token)
	i=i+1
	print(i)

#preProcessedData = preprocess_documents(f_reader)
trainedModel = Word2Vec(preProcessedData, size=100, window=5, min_count=5, workers=8)
trainedModel.save("word2vec.model")

