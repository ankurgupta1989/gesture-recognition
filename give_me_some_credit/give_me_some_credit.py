__author__ = 'ankurgupta'
# Imports
import random
from io import open
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn import cross_validation
import numpy as np
SEED = 42  # always use a seed for randomized procedures

np.set_printoptions(precision=3)

def get_mean(input_file, limitInput):
    # Calculate the mean.
    fh = open(input_file, 'rt')
    line = fh.readline()
    total = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totalCount = 0
    while line and totalCount<=limitInput:
        line = line.replace('\n','')
        values = line.split(',')
        values = values[2:]

        for i in range(10):
            value = values[i]
            if value != "NA":
                total[i] += float(value)
                count[i] += 1
        line = fh.readline()
        totalCount += 1

    for i in range(10):
        total[i] = total[i]/count[i]

def getFeatures(input_file, isTraining, limitInput):
    fh = open(input_file, 'rt')
    line = fh.readline()
    count = 0
    reject = 0
    pos_count = 0
    neg_count = 0
    features = []
    replaceCount = 0
    while line and count<=limitInput:
        line = line.replace('\n','')
        values = line.split(',')

        # For the test file, target will be unspecified.
        if not isTraining:
            values[1] = 1

        # Classify as positive and negative.
        target = int(values[1])

        if target==1:
            pos_count += 1
            target = 1
        else:
            neg_count += 1
            target = 0

        # Put the input features in the array
        values = values[2:]

        new_values = []
        for i in range(10):
            value = values[i]
            if value == "NA":
                # Use -1 in the value.
                value = -1
            if i==2 and (value=="96" or value=="98"):
                value = -1
                replaceCount += 1

            new_values.append(float(value))
        new_values = new_values

        # Increasing the size of positives.
        if isTraining and target==1:
            for i in range(14):
                features.append((new_values, target))
            pos_count += 13
        else:
            features.append((new_values, target))

        line = fh.readline()
        count += 1

    print "Replace count " + str(replaceCount)
    print "Reject number " + str(reject)
    print "Count number " + str(count)
    print "Positive count number " + str(pos_count)
    print "Negative count number " + str(neg_count)

    # Don't shuffle for test data.
    if isTraining:
       random.shuffle(features)

    # Take only few features for classification.
    if isTraining:
        featureCount = 60000
        print "Considering " + str(featureCount) + " features"
        features = features[:featureCount]

    targets = []
    inputF = []
    for (feature, target) in features:
        inputF.append(feature)
        targets.append(target)

    return (inputF, targets)



def write_to_file(output_file, predictions):
    fw = open(output_file, 'w')
    cnt = 1
    for pred in predictions:
        fw.write(str(cnt) + "," + unicode(pred) + "\n")
        cnt += 1
    fw.close()

input_file = '/Users/ankurgupta/Desktop/AI-Project-Kaggle/cs-training.csv'
(train_features, train_targets) = getFeatures(input_file, True, 10000000)
#print "Training features " + str(train_features)
#print "Training targets " + str(train_targets)


# Create a classifier
#classifier = svm.SVC(C=1e-2)
#classifier = linear_model.LogisticRegression()
classifier = ensemble.RandomForestClassifier(n_estimators=1800, criterion='gini', max_depth=None,
                                            min_samples_split=2, min_samples_leaf=1, max_features='auto',
                                            max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1,
                                            random_state=None, verbose=0)
#classifier = naive_bayes.GaussianNB()
#classifier = tree.DecisionTreeClassifier()
# classifier = neighbors.KNeighborsClassifier(n_neighbors=200)

# Using cross-validation to improve the score.
mean_auc = 0.0
num_points = 10
for i in range(num_points):
    # Hold 20% of the data as test set.
    X_train_dev, X_test_dev, Y_train_dev, Y_test_dev = cross_validation.train_test_split(train_features, train_targets, test_size=0.20, random_state=i*SEED)
    print "Train dev length " + str(len(X_train_dev))
    print "Test dev length " + str(len(X_test_dev))
    print "Train dev class " + str(len(Y_train_dev))
    print "Test dev class " + str(len(Y_test_dev))

    classifier.fit(X_train_dev, Y_train_dev)
    print "Training complete"
    preds = classifier.predict_proba(X_test_dev)[:,1]
    print "Prediction done"
    # Compute AUC metrics for this cross validation fold.
    print "y true " + str(Y_test_dev)
    print "y score " + str(preds)
    # fpr, tpr, thresholds = metrics.roc_curve(Y_test_dev, preds)
    # print "fpr " + str(fpr)
    # print "tpr " + str(tpr)
    # print "thresholds " + str(thresholds)
    # print "Metrics roc curve complete"
    # roc_auc = metrics.auc(fpr, tpr)
    roc_auc = metrics.roc_auc_score(Y_test_dev, preds)
    print "AUC (fold %d/%d): %f" % (i + 1, num_points, roc_auc)
    mean_auc += roc_auc

print "Mean AUC: %f" % (mean_auc/num_points)

classifier.fit(train_features, train_targets)


input_file = '/Users/ankurgupta/Desktop/AI-Project-Kaggle/cs-test.csv'
(features, targets) = getFeatures(input_file, False, 10000000)
#print "Testing features " + str(features)
#print "Testing targets " + str(targets)

# predicted = classifier.predict(features) #This is for the support vector machine.
predicted = classifier.predict_proba(features)
new_predicted = []
for predict in predicted:
    #temp = predict #This is for the support vector machines.
    temp = round(predict[1], 8)
    new_predicted.append(temp)

output_file = '/Users/ankurgupta/Desktop/AI-Project-Kaggle/prediction.csv'
write_to_file(output_file, new_predicted)



# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
