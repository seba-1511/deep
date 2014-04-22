__author__ = 'gabrielpereyra'

# Add path to nltk
from collections import defaultdict
import sys
sys.path.append("/Users/gabrielpereyra/anaconda/lib/python2.7/site-packages")
import nltk
import string

# create test/train corpora from brown corpus
brown = nltk.corpus.brown
corpus = [word.lower() for word in brown.words()]
split = 95*len(corpus)/100
train = corpus[:split]
test = corpus[split:]

# nltk ngram module results

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print word,
        word = cfdist[word].max()


text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

print cfd['Living']
print generate_model(cfd, 'living')