__author__ = 'ankurgupta'

import csv
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
import random

SEED = 42 # always use a seed for randomized process.

def modify(dictList, val, index, is_train):
    enum_conversion_index_list = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 26, 27, 28, 29, 30]
    if not is_train:
        enum_conversion_index_list = [x - 1 for x in enum_conversion_index_list]

    if (is_train and index == 2) or (not is_train and index==1):
        return (datetime.datetime.strptime(val, "%m/%d/%Y") - datetime.datetime(1970, 1, 1)).total_seconds()

    elif index in enum_conversion_index_list:
        dict = dictList[index]
        if dict.get(val, 0) == 0:
            dict[val] = max(dict.values()) + 1
        return dict[val]

    else:
        if val == 'NULL':
            return 0
        else:
            return float(val)


def get_features(input_file, is_train, limit):
    f = open(input_file)
    csv_f = csv.reader(f)
    csv_f.next()
    features = []
    classes = []
    tuple = []
    totalFeatures = 33
    if is_train:
        start = 2
        end = 34
    else:
        start = 1
        end = 33


    dictList = []
    for i in range(totalFeatures + 1):
        dictList.append({'RANDOM': 0})


    cnt = 0
    pos_count = 0
    neg_count = 0
    order = {}
    while True and cnt <= limit:
        try:
            list = csv_f.next()
            row = []
            for index in range(start, end):
                row.append(modify(dictList, list[index], index, is_train))

            output_class = list[1]
            if is_train and output_class == '1':
                # prob = random.random()
                # if prob < 0.15:
                #     tuple.append((row, output_class))
                    # features.append(row)
                    # classes.append(output_class)
                for i in range(7):
                    tuple.append((row, output_class))
                #     features.append(row)
                #     classes.append(output_class)
            else:
                tuple.append((row, output_class))
                # features.append(row)
                # classes.append(output_class)

            order[cnt] = list[0]
            cnt += 1
        except StopIteration:
            break

    f.close()

    if is_train:
        random.shuffle(tuple)

    for (feature, output_class) in tuple:
        features.append(feature)
        if is_train:
            classes.append(int(output_class))
        else:
            classes.append(None)

    total_cnt = 0
    for i in range(len(features)):
        output_class = classes[i]
        if output_class == '1':
            pos_count += 1
        else:
            neg_count += 1
        total_cnt += 1

    print "Total number of entries " + str(total_cnt)
    print "Total positive entries " + str(pos_count)
    print "Total negative entries " + str(neg_count)

    return features, classes, order


def get_dict(list):
    dict = {}
    for val in list:
        if dict.get(val, 0) == 0:
            dict[val] = 1
        else:
            dict[val] += 1
    return dict


def write_to_file(output_file, predicted, order):
    f = open(output_file, 'w')
    f.write("RefId,IsBadBuy\n")
    cnt_start = 0
    for predict in predicted:
        f.write(str(order[cnt_start]) + "," + str(predict) + "\n")
        cnt_start += 1


def prune_features(features, use_list):
    new_features = []
    for feature in features:
        row = []
        cnt = 0
        for to_use in use_list:
            if to_use:
                row.append(feature[cnt])
            cnt += 1
        new_features.append(row)
    return new_features

def normalize(features):
    return preprocessing.normalize(features)

def standardize(features):
    return preprocessing.scale(features)

input_file = '/Users/ankurgupta/Desktop/AI-Project-Kaggle/DontGetKicked/dont-get-kicked-training.csv'
features, classes, train_order = get_features(input_file, True, 100000000)


# model = LogisticRegression(C=1e+10)
# model = SVC(C=1e+10)
# model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
#                                             min_samples_split=2, min_samples_leaf=1, max_features='auto',
#                                             max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1,
#                                             random_state=None, verbose=0)
# model = naive_bayes.GaussianNB()
model = naive_bayes.MultinomialNB()
# model = neighbors.KNeighborsClassifier(n_neighbors=50)
# model = DecisionTreeClassifier()
# model = Perceptron()
# model = GradientBoostingClassifier()


# Take num_rows entries
num_rows = 120000
features = features[:num_rows]
classes = classes[:num_rows]


# Use RFE.
rfe = RFE(model, 28)
rfw = rfe.fit(features, classes)
usage = rfe.support_
print(usage)
print(rfe.ranking_)

features = prune_features(features, usage)
# features = standardize(normalize(features))

num_folds = 5.0

print 'Running code for cross validation using train_test_split'
for i in range(int(num_folds)):
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(features, classes, test_size=1/num_folds, random_state=i*SEED)
    model.fit(X_train, Y_train)
    predicted_dev = model.predict(X_test)
    print metrics.roc_auc_score(Y_test, predicted_dev)


print 'Running code for cross validation using cross_val_score'
kfold = cross_validation.KFold(n=len(features), n_folds=num_folds)
results = cross_validation.cross_val_score(model, features, classes, cv=kfold, scoring='roc_auc')
print(results)

model.fit(features, classes)

input_file = '/Users/ankurgupta/Desktop/AI-Project-Kaggle/DontGetKicked/dont-get-kicked-test.csv'
features_test, dummy, order = get_features(input_file, False, 100000000)

features_test = prune_features(features_test, usage)
# features_test = standardize(normalize(features_test))

predicted = model.predict(features_test)

print get_dict(predicted)
output_file = '/Users/ankurgupta/Desktop/AI-Project-Kaggle/DontGetKicked/prediction.csv'
write_to_file(output_file, predicted, order)


