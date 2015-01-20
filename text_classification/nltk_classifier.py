__author__ = 'ankurgupta'
import json
import nltk
from nltk import word_tokenize

directory = '/Users/ankurgupta/Desktop/AI-Project/yelp_dataset_challenge_academic_dataset/'
file_name_test = directory + 'review_test.json'
file_name_train = directory + 'review_train.json'
file_name = directory + 'yelp_academic_dataset_review.json'

fh = open(file_name, 'rt')
line = fh.readline()
train = []
tokens = []
num = 10000
for i in range(num):
    data = json.loads(line)
    train.append((data["text"], data["stars"]))
    tokens = tokens + word_tokenize(data["text"])
    line = fh.readline()

fh.close()

print "Tokens length " + str(len(tokens))

all_words = nltk.FreqDist(w.lower() for w in tokens)
word_features = all_words.keys()[:2000]


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in train]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
