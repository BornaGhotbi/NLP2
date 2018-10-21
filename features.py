import nltk
import re
from textblob import TextBlob

tweet={}
tweet['text']="sF is noT a shit good girl?"

def punctuationanalysis(tw):
    hasqmark = 0
    if tw['text'].find('?') >= 0:
        hasqmark = 1
    hasemark = 0
    if tw['text'].find('!') >= 0:
        hasemark = 1
    hasperiod = 0
    if tw['text'].find('.') >= 0:
        hasperiod = 0

    return hasqmark,hasemark,hasperiod

def negationwordcount(tw):
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', tw['text'].lower()))
    countnegation = 0
    negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
                     'neither', 'nor', 'nowhere', 'hardly',
                     'scarcely', 'barely', 'don', 'isn', 'wasn',
                     'shouldn', 'wouldn', 'couldn', 'doesn', 'misinform','fake','rumour']
    for negationword in negationwords:
        if negationword in tokens:
            countnegation += 1
    return countnegation

def swearwordcount(tw):
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', tw['text'].lower()))
    swearwords = []
    with open('badwords.txt', 'r') as f:
        for line in f:
            swearwords.append(line.strip().lower())

    hasswearwords = 0
    for token in tokens:
        if token in swearwords:
            hasswearwords += 1

    return hasswearwords

def capitalratio(tw):
    uppers = [l for l in tw['text'] if l.isupper()]
    capitalratio = len(uppers) / len(tw['text'])
    return capitalratio

def contentlength(tw):
    charcount=0
    for word in tw['text']:
        charcount+=len(nltk.word_tokenize(word))
    wordcount = len(nltk.word_tokenize(tw['text']))
    return charcount,wordcount

def poscount(tw):
    postag = []
    poscount = {}
    word_tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', tw['text'].lower()))
    for word in word_tokens:
        postag = nltk.pos_tag(word)
        for g1 in postag:
            if g1[1] not in poscount:
                poscount[g1[1]] = 1
            else:
                poscount[g1[1]] += 1
    return poscount

def supportwordcount(tw):
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', tw['text'].lower()))
    countsupport = 0
    supportwords = ['yes', 'yeah', 'ya', 'agree', 'support', 'suck',
                     'sucks', 'holy', 'possible', 'aha',
                     'since', 'accurate', 'bad', 'oh', 'agree',
                     'confirm', 'definitely', 'acoording', 'official']
    for supportword in supportwords:
        if supportword in tokens:
            countsupport += 1
    return countsupport

def sentimentscore(tw):
        analysis = TextBlob(tw['text'])
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

if __name__ == "__main__":

    hasperiod=punctuationanalysis(tweet)
    negationwordcount(tweet)
    swearwordcount(tweet)
    capitalratio(tweet)
    contentlength(tweet)
    poscount(tweet)
    supportwordcount(tweet)
    sentimentscore(tweet)
