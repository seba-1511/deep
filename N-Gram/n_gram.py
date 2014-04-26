import sys
sys.path.append("/Users/gabrielpereyra/anaconda/lib/python2.7/site-packages")

from nltk import NgramModel
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist

if __name__ == '__main__':

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