import collections, sys, os
import nltk
import pandas
from textblob import TextBlob
#optionally, pass in prompt for prompt features
#treating prompts and text as STRINGS, not lists of words
def featureExtractor(text, prompt=""):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    promptWords = nltk.word_tokenize(prompt)
    partsOfSpeech = nltk.pos_tag(words)
    blob = TextBlob(text)
    textList = text.split()
    autoCorrectList = blob.correct().split()

    features = collections.defaultdict(float)

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
	
def learnPredictor(trainExamples, testExamples, featuresList, numIters, eta):
    weights = {} 
    for _ in range(numIters):
        for i in range(len(trainExamples)):
            score=trainExamples[i]
            feats = featuresList[i]
            for key in feats:
                gradient = 2*feats[key]*(weights.get(key,0.0) * feats[key] - score)
                weights[key]=weights.get(key,0.0)-eta*gradient
    return weights

df = pandas.read_excel('training_set_essay1_100.xlsx', sheet_name='training_set')
prompt ="More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."
id = df['essay_id']
essays = df['essay']
scoreOne = df['rater1_domain1']
scoreTwo = df['rater2_domain1']
#scoreThree = df['rater3_domain1']

total = 0
featuresList = []
trainExamples = []
for i in range(0, 10):
    featuresList.append(featureExtractor(essays[i], prompt))
    trainExamples.append((scoreOne[i]+scoreTwo[i])/2.0)
testExamples = {}

numIters = 500
eta = 0.05
print("Starting learning")
learnedWeights = learnPredictor(trainExamples, testExamples, featuresList, numIters, eta)
print("learned weights with " + str(numIters))
print(learnedWeights)

def dotProd(v1, v2):
    total = 0
    for key in v1:
        if key in v2 and v2[key] != float('nan'):
            total += v1[key]*v2[key]
    return total

for i in range (0, 10):
    print(i, dotProd(featuresList[i], learnedWeights))