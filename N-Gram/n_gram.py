import sys
sys.path.append("/Users/gabrielpereyra/anaconda/lib/python2.7/site-packages")

from collections import defaultdict
from math import log

from nltk import NgramModel
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist

class unigram_model():

    def __init__(self):
        self.cfd = defaultdict(int)

        for sent in brown.sents()[0:100]:
            for word in sent:
                self.cfd[word] += 1

        print self.cfd

    # unigram probability
    # frequency / total words (0 if word not found)
    def prob(self, word):
        if word in self.cfd:
            return self.cfd[word] / float(len(self.cfd))
        else:
            return 0

    def logprob(self, word):

        return -log(self.prob(word), 2)

    def generate(self, num_words):
        text = []
        for i in range(num_words):
            text.append('.')
            #append most frequent word

    def entropy(self, text):
        e = 0.0
        for i in range(len(text)):
            token = text[i]
            e += self.logprob(token)
        return e / float(len(text))

    def perplexity(self, text):
        return pow(2.0, self.entropy(text))


if __name__ == '__main__':

    """
    # load nltk ngram model
    est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
    lm = NgramModel(2, brown.words(categories='news'), estimator=est)

    # functions to implement
    print lm.prob('County',['The'])
    print lm.logprob('County',['The'])
    print lm.choose_random_word(['The'])
    print lm.generate(10)
    print lm.entropy(brown.words()[0:100])
    print lm.perplexity(brown.words()[0:100])
    """

    unigram = unigram_model()

    print unigram.logprob('all')
    print unigram.logprob('welfare')
    print unigram.entropy(brown.words()[:100])
    print unigram.perplexity(brown.words()[:100])