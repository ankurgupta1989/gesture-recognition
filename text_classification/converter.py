__author__ = 'ankurgupta'

import json
import io
from io import open

directory = '/Users/ankurgupta/Desktop/AI-Project/yelp_dataset_challenge_academic_dataset/'
file_name = directory + 'yelp_academic_dataset_review.json'

total_records = 10000

train_pos = directory + 'yelp-train-pos.txt'
train_neg = directory + 'yelp-train-neg.txt'

def convert(input_file, new_train, new_test):
    fh = open(input_file, 'rt', encoding='utf8')
    fPos = open(new_train, 'w', encoding='utf8')
    fNeg = open(new_test, 'w', encoding='utf8')
    line = fh.readline()
    posCount = 0
    negCount = 0
    while line:
        if posCount >= total_records and negCount >= total_records:
            break
        data = json.loads(line.decode('utf8'))
        temp = unicode(data["text"])
        temp = temp.replace('\n', ' ')
        temp = temp.replace('\r', ' ')
        temp = temp.replace('\t', ' ')
        temp = temp.replace(u'\377', '')
        if data["stars"] == 5:
            if posCount >= total_records:
                line = fh.readline()
                continue
            fPos.write(temp + "\n")
            posCount += 1
        if data["stars"] == 1:
            if negCount >= total_records:
                line = fh.readline()
                continue
            fNeg.write(temp + "\n")
            negCount += 1

        line = fh.readline()

    fh.close()
    fPos.close()
    fNeg.close()
    print posCount
    print negCount
convert(file_name, train_pos, train_neg)