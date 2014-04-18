__author__ = 'gabrielpereyra'

from collections import defaultdict

def word_count():

    f = open('gene.train','r')
    lines = [line.split() for line in f.readlines()]
    lines = filter(None, lines) # removes empty lists

    wordCountDict = defaultdict(int)

    for line in lines:
        wordCountDict[tuple(line)] += 1

    f.close()
    return wordCountDict

def one_gram(wordCountDict):

    I_GENE = 0
    ZERO   = 0

    for word in wordCountDict:
        if word[1] == 'I-GENE':
            I_GENE += wordCountDict[word]
        else:
            ZERO += wordCountDict[word]

    print ZERO
    print I_GENE

if __name__ == '__main__':

    wordCountDict = word_count()

    #for word in wordCountDict:
    #   print wordCountDict[word], "WORDTAG", word[1], word[0]

    one_gram(word_count())