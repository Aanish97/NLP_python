from gensim.models import Word2Vec
from nltk.stem import PorterStemmer

ps = PorterStemmer()
trainedModel = Word2Vec.load("word2vec.model")

words = ['clean', 'unclean', 'amazed', 'friendly']

stemmed_words= list(map(PorterStemmer().stem, words))

for w in stemmed_words:
    print("Similar words to '" + str(w) + "' :")
    print(trainedModel.most_similar(positive=w))
