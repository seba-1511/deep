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

def n_gram(n):

    f = open('gene.train','r')
    lines = [line.split() for line in f.readlines()]

    sentence = []
    sentenceList = []
    oneGramDict = defaultdict(int)

    for line in lines:

        if line != []:
            sentence.append(line)
        else:
            sentence.insert(0,['Start','*'])
            sentence.append(['End','STOP'])
            sentenceList.append(sentence)
            sentence = []

    for sentence in sentenceList:

        currentGram = []

        for word in sentence:
            currentGram.append(word[1])

            if len(currentGram) == n:
                oneGramDict[tuple(currentGram)] += 1
                currentGram.pop(0)

    for word in oneGramDict:

        print oneGramDict[word], "1-GRAM", word

if __name__ == '__main__':

    wordCountDict = word_count()

    #for word in wordCountDict:
    #   print wordCountDict[word], "WORDTAG", word[1], word[0]

    n_gram(1)