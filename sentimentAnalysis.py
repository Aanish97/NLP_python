from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import pandas as pd
import csv

path_data = "D:/.Semester 7/IR/Assignments/assignment 3/Datasets/Question2 Dataset.tsv"

with open(path_data, "r", encoding='ISO-8859-1') as tsvfile:

  reader = csv.reader(tsvfile, delimiter='\t')

  df = pd.DataFrame(reader, columns = ['id', 'sentiment', 'review'])

  df['review'] = df['review'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

  X_train, X_test = train_test_split(df)

  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(X_train['review'])
  X_i = vectorizer.transform(X_test['review'])

  transformer = TfidfTransformer()
  Y = transformer.fit_transform(X)
  Y_i = transformer.transform(X_i)


  clf = MultinomialNB().fit(X, X_train['sentiment'])
  X_prediction = clf.predict(X_i)
  Y_prediction = clf.predict(Y_i)

  raw_count = accuracy_score(X_test['sentiment'], X_prediction)
  tfIDF = accuracy_score(X_test['sentiment'], Y_prediction)

  print("accuracy of raw count is " + str(raw_count*100) + "%")
  print("accuracy of tf IDF is " + str(tfIDF*100) + "%")

