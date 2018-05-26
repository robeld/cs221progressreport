import pandas

import collections, sys, os
#pip install -U nltk
import nltk
#pip install -U textblob
#python -m textblob.download_corpora
from textblob import TextBlob

def featureExtractor(text, prompt=""):
	sentences = nltk.sent_tokenize(text)
	words = nltk.word_tokenize(text)
	promptWords = nltk.word_tokenize(prompt)
	partsOfSpeech = nltk.pos_tag(words)
	blob = TextBlob(text)
	textList = text.split()
	autoCorrectList = blob.correct().split()

	features = collections.defaultdict(int)

	features["polarity"] = blob.sentiment.polarity
	features["subjectivity"] = blob.sentiment.subjectivity
	features["unique words"] = len(set(words))
	features["word count"] = len(words)
	features["sentence count"] = len(sentences)

	for index, elem in enumerate(textList):
		if autoCorrectList[index] != elem:
			features["misspelled words proportion"] += 1/len(textList)

	features["prompt shared words proportion"] = len(set(words) & set(promptWords))/features["word count"]
	features["average sentence length"] = features["word count"]/features["sentence count"]
	features["average word length"] = sum(len(x) for x in words)/features["word count"]
	meanOfSquare = (sum(len(x)**2 for x in sentences))/len(sentences)
	features["sentence length variance"] = meanOfSquare - (features["average sentence length"]**2)
	features["passive verb proportion"] = (text.count("is") + text.count("was") + text.count("were") + text.count("had") + text.count("have") + text.count("been"))/features["word count"]

	for elem in partsOfSpeech:
		features[elem[1]] += (1/features["word count"])

	return features


df = pandas.read_excel('training_set_rel3.xlsx', sheet_name='training_set')
id = df['essay_set']
essays = df['essay']
scoreOne = df['rater1_domain1']
scoreTwo = df['rater2_domain1']
scoreThree = df['rater3_domain1']


featureScore = []
#total = 0
#for i in range(0, len(essays)):
#    if id[i] == 1:
#        total+=1
for i in range(0, 200):#int(2*total/3)):
    featureScore.append((i,featureExtractor(essays[i])))

#print(total)
#print(id[1])

def dist(one, two):
    sum = 0
    for key in one:
    	if one[key] != 0:
        	sum+= abs( (one[key]-two[key])/float(one[key]))
    return sum

def closestScore(text):
    feature = featureExtractor(text)
    currMin = float("inf")
    index = -1
    for i in range(0,200):
        distance = dist(feature, featureScore[i][1])
        if distance < currMin:
            index = i
            currMin = distance
    return (scoreOne[index] + scoreTwo[index])/2.0 #+ scoreThree[index])/3.0

#Above I've created all the featuresScores, here I now need to test meaning


countRight = 0
sqLoss = 0
file = open("scoreResult.txt", "w")
for i in range(200, 300):
	print(i)
	guess = closestScore(essays[i])
	if((scoreOne[i] + scoreTwo[i])/2.0 == guess):
		countRight+=1
	sqLoss += (guess - (scoreOne[i] + scoreTwo[i])/2.0)**2
	file.write("Actual Score: " + str((scoreOne[i] + scoreTwo[i])/2.0) + " Guessed Score: " + str(guess))
	file.write("\n")
file.close()
#print(essays[1])
print(countRight)
print(sqLoss/100.0)