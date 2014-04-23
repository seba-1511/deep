import sys
sys.path.append("/Users/gabrielpereyra/anaconda/lib/python2.7/site-packages")
import nltk

f = open('reuters21578/reut2-000.sgm','r')

text = f.readlines()[:100]

for line in text:
    print line, line.find('BODY')
    