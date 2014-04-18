__author__ = 'gabrielpereyra'

from collections import defaultdict

def word_count():

    f = open('gene.train','r')
    lines = [line.split() for line in f.readlines()]
    lines = filter(None, lines) # removes empty lists

    wordCountDict = defaultdict(int)

    for line in lines:
        wordCountDict[tuple(line)] += 1

    for word in wordCountDict:
        print wordCountDict[word], "WORDTAG", word[1], word[0]


if __name__ == '__main__':
    word_count()