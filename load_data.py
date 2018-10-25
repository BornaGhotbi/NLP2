import os
import json
import codecs
import gensim
from nltk.corpus import stopwords as stp
import numpy as np
import nltk
import re
from textblob import TextBlob


global model
stop_words = set(stp.words('english'))
tweets=dict()
punctuations= ["\"","(",")","*",",","-","_",".","~","%","^","&","!","#"
               "=","\'","\\","+","/",":","[","]","«","»","،","؛","?",".","…","$",
               "|","{","}","٫",";",">","<","1","2","3","4","5","6","7","8","9","0"]



def load_targets():
    filepath_source = './traindev/rumoureval-subtaskA-train.json'
    train = json.load(codecs.open(filepath_source, 'r', 'utf-8-sig'))
    return train

def class2num(str):
    if str=="comment":
        return 1
    elif str== "support":
        return 2
    elif str=="query":
        return 3
    elif str=="deny":
        return 4


def load_google_vector():
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model


def load_dataset():
    num_tweets=0
    for filename1 in os.listdir("./rumoureval-data"):
        if filename1!=".DS_Store":
            for filename2 in os.listdir("./rumoureval-data/"+filename1):
                if filename2!=".DS_Store":
                    filepath_source="./rumoureval-data/"+filename1+"/"+filename2+"/source-tweet/"+filename2+".json"
                    tweets[filename2]=load_tweet(filepath_source,filename2)
                    replies=[]
                    num_tweets+=1
                    for filename3 in os.listdir("./rumoureval-data/"+filename1+"/"+filename2+"/replies"):
                        filepath_reply="./rumoureval-data/"+filename1+"/"+filename2+"/replies/"+filename3
                        tweets[filename3[0:filename3.find(".")]]=load_tweet(filepath_reply,filename2)
                        replies.append(filename3[0:filename3.find(".")])
                        num_tweets+=1
                    tweets[filename2]["replies"]=replies
                    tweets[filename2]["structure"]= json.load(codecs.open("./rumoureval-data/"+filename1+"/"+filename2+"/structure.json", 'r', 'utf-8-sig'))
    return num_tweets


def load_tweet(path,source_id):
    global model
    tweet_raw = json.load(codecs.open(path, 'r', 'utf-8-sig'))
    conversation = dict()
    conversation["id"] = str(tweet_raw["id"])
    conversation["raw_text"]=tweet_raw["text"]
    tmp=conversation["raw_text"].lower()
    words=[]
    for word in tmp.split():
        if word.startswith( 'http' ):
            conversation["url"]=word
        else:
            conversation["url"]=None
            for pt in punctuations:
                word=word.replace(pt,"")
            if  word in model and word not in stop_words and word != "":
                words.append(word)

    conversation["text"] = tmp
    conversation["words"]=words
    conversation["source_id"] = str(source_id)
    conversation["reply_to"] = str(tweet_raw["in_reply_to_status_id"])
    conversation["vector"]=tweet2v(conversation)
    conversation["num_followers"]=tweet_raw["user"]["followers_count"]
    return conversation

#vector
def tweet2v(conversation):
    global model
    num_features = 300
    temp_rep = np.zeros(num_features)
    if len(conversation["words"])!=0:
        for word in conversation["words"]:
            temp_rep += model[word]
    return temp_rep/len(conversation["words"])

#is reply
def is_reply(conversation):
    if conversation["reply_to"]==None:
        return 0
    return 1

#similarity with other tweets
def relation2other(conversation):
    similarity2source=0
    similarity2reply2=0
    similarity2others=0

    words1=conversation["words"]
    words2=tweets[conversation["source_id"]]["words"]
    if len(words1) > 0 and len(words2) > 0:
        similarity2source= model.n_similarity(words1, words2)

    words2=tweets[conversation["reply_to"]]["words"]
    if len(words1) > 0 and len(words2) > 0:
        similarity2reply2= model.n_similarity(words1, words2)

    words2=[]
    for tweet_id in tweets[conversation["source_id"]]["replies"]:
        words2.append(tweets[tweet_id])
    flatten = lambda words2: [item for sublist in words2 for item in sublist]
    if len(words1) > 0 and len(flatten) > 0:
        similarity2others= model.n_similarity(words1, flatten)
    return [similarity2reply2,similarity2others,similarity2source]

def has_url(conversation):
    if conversation["url"]==None:
        return 0
    return 1

#punctuations
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
    uppers = [l for l in tw['raw_text'] if l.isupper()]
    capitalratio = len(uppers) / len(tw['raw_text'])
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
    return poscount.values()

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
        return analysis.sentiment.polarity


def tweet2features():
    for tweet_id in tweets:
        conversation=tweets[tweet_id]
        features=[]
        list=conversation["vector"]
        for item in list:
            if item!=item:
                features.append(0)
            else:
                features.append(item)
        features.append(is_reply(conversation))
        # features.append(relation2other(conversation))
        list =punctuationanalysis(conversation)
        for item in list:
            features.append(item)
        features.append(has_url(conversation))
        features.append(negationwordcount(conversation))
        features.append(swearwordcount(conversation))
        features.append(capitalratio(conversation))
        list = contentlength(conversation)
        for item in list:
            features.append(item)
        # list=poscount(conversation)
        # for item in list:
        #     features.append(item)
        features.append(supportwordcount(conversation))
        features.append(sentimentscore(conversation))
        features.append(conversation["num_followers"])
        # print("features",len(features),features)
        conversation["features"]= features
        tweets[conversation["id"]]=conversation

branch_array=[]
target_array=[]
max_depth = 0



def build_branches():
    for tweet in tweets.values():
        if "structure" in tweet.keys() :
            if tweet["id"] in train:
                label=class2num(train[tweet["id"]])
            else:
                label=1
            build_branch4tweet(tweet["structure"][tweet["id"]],[tweet["features"]], [label],1)


def build_branch4tweet(structure,array, label_array,branch_depth):
    global max_depth
    if len(structure)==0:
        branch_array.append(array)
        target_array.append(label_array)
        if max_depth<branch_depth:
            max_depth=branch_depth

    else:
        for id in structure.keys():
            if id in tweets.keys():
                array.append(tweets[id]["features"])
                if id in train:
                    label = class2num(train[id])
                else:
                    label = 1
                label_array.append(label)
                build_branch4tweet(structure[id],array,label_array,branch_depth+1)
            else:
                branch_array.append(array)
                target_array.append(label_array)
                if max_depth < branch_depth:
                    max_depth = branch_depth




##### run
model= load_google_vector()
num_tweets=load_dataset()
tweet2features()
train = load_targets()
build_branches()

##Branch array: array that includes branches, each branch include tweets that are Vectors(1,313)
##target array: array that include label of branches [1,2,3,4]
print("array branch",len(branch_array),len(branch_array[1]),len(branch_array[1][0]))
print("array target",len(target_array),len(target_array[1]))
print("max branch depth",max_depth)





