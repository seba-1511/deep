import sys
sys.path.append("/Users/gabrielpereyra/anaconda/lib/python2.7/site-packages")
import nltk

def tokenize_sent(text):
    sent_detector = nltk.load('tokenizers/punkt/english.pickle')

    return sent_detector.tokenize(text.strip())

def extract_body_text(raw_text):

    body_text = []
    current_body = []
    in_body = False

    for line in raw_text:

        if in_body == True:
            if line.find('</BODY>') > 0:
                end_index = line.find('<BODY>')
                current_body.append(line[:end_index])
                body_text.append(current_body[:-2]) # -2 to strip out "Reuter" and "&#3;</BODY></TEXT>"
                current_body = []
                in_body = False
            else:
                current_body.append(line)

        if in_body == False:
            if line.find('<BODY>') > 0:
                start_index = line.find('<BODY>')
                current_body.append(line[start_index+len('<BODY>'):])
                in_body = True

    return body_text

if __name__ == '__main__':

    f = open('reuters21578/reut2-000.sgm','r')

    raw_text = f.readlines()[:100]

    text = extract_body_text(raw_text)

    sentences = tokenize_sent(' '.join(text[0]))

    print sentences[0]