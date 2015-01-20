__author__ = 'ankurgupta'

import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


SEED = 42
# Helper functions..

def get_maximum():
    return 1000000000000

def get_minimum():
    return -1000000000000

def distance(first_point, second_point):
    delta_X = second_point[0] - first_point[0]
    delta_Y = second_point[1] - first_point[1]
    return max(0.1, math.sqrt(delta_X*delta_X + delta_Y*delta_Y))

def angle(p1, p2, p3):
    dx = p3[0] - p2[0]
    dy = p3[1] - p2[1]

    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]

    num = dx*dy1 + dx1*dy
    den = max(0.1, dx*dx1 - dy*dy1)

    return math.atan(num/den)

def get_bounding_box(points):
    xmin =  get_maximum()
    ymin =  get_maximum()
    xmax = get_minimum()
    ymax = get_minimum()
    for point in points:
        x = point[0]
        y = point[1]
        if x > xmax:
            xmax= x
        if y > ymax:
            ymax = y
        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y
    return (xmin, ymin, xmax, ymax)

def get_first_and_last(points):
    xfirst = points[0][0]
    yfirst = points[0][1]
    p_len = len(points)
    xlast = points[p_len - 1][0]
    ylast = points[p_len - 1][1]
    return (xfirst, yfirst, xlast, ylast)

# ----------------------------------------------------------------------------------------------------------------------
# Features.

# sin A
def get_feature_1(points):
    first_point = points[0]
    second_point = points[4]
    f1 = (second_point[1] - first_point[1])/distance(first_point, second_point)
    return f1

# cos A
def get_feature_2(points):
    first_point = points[0]
    second_point = points[4]
    f2 = (second_point[0] - first_point[1])/distance(first_point, second_point)
    return f2

# bounding box diagonal.
def get_feature_3(points):
    (xmin, ymin, xmax, ymax) = get_bounding_box(points)
    return distance([xmin, ymin], [xmax, ymax])

# bounding box angle.
def get_feature_4(points):
    (xmin, ymin, xmax, ymax) = get_bounding_box(points)
    val = (ymax - ymin)/max(0.1, (xmax - xmin))
    return math.atan(val)

# distance between first and last points.
def get_feature_5(points):
    (xfirst, yfirst, xlast, ylast) = get_first_and_last(points)
    return distance([xfirst, yfirst], [xlast, ylast])

# cos B
def get_feature_6(points):
    (xfirst, yfirst, xlast, ylast) = get_first_and_last(points)
    temp = xlast - xfirst
    return temp/distance([xfirst, yfirst], [xlast, ylast])

# sin B
def get_feature_7(points):
    (xfirst, yfirst, xlast, ylast) = get_first_and_last(points)
    temp = ylast - yfirst
    return temp/ distance([xfirst, yfirst], [xlast, ylast])

# total stroke length.
def get_feature_8(points):
    p_len = len(points)
    dist = 0
    for i in range(p_len - 1):
        dist += distance(points[i], points[i+1])
    return dist

# angle summation.
def get_feature_9(points):
    p_len = len(points)
    total_angle = 0
    for i in range(p_len - 2):
        a = angle(points[i], points[i+1], points[i+2])
        total_angle += a
    return total_angle

# angle summation absolute.
def get_feature_10(points):
    p_len = len(points)
    a = 0
    for i in range(p_len - 2):
        a += math.fabs(angle(points[i], points[i+1], points[i+2]))
    return a

# angle summation square.
def get_feature_11(points):
    p_len = len(points)
    total_angle = 0
    for i in range(p_len - 2):
        a = angle(points[i], points[i+1], points[i+2])
        total_angle += a*a
    return total_angle

# max speed.
def get_feature_12(points):
    p_len = len(points)
    val = get_minimum()
    for i in range(p_len - 1):
        val = distance(points[i], points[i+1])
        dist_square = val*val
        time_square = max(0.1, math.sqrt(points[i+1][2] - points[i][2]))
        val = max(val, dist_square/time_square)
    return val

# total time.
def get_feature_13(points):
    p_len = len(points)
    return points[p_len - 1][2] - points[0][2]


def make_features(points):
    feature = []
    feature.append(get_feature_1(points))
    feature.append(get_feature_2(points))
    feature.append(get_feature_3(points))
    feature.append(get_feature_4(points))
    feature.append(get_feature_5(points))
    feature.append(get_feature_6(points))
    feature.append(get_feature_7(points))
    feature.append(get_feature_8(points))
    feature.append(get_feature_9(points))
    feature.append(get_feature_10(points))
    feature.append(get_feature_11(points))
    feature.append(get_feature_12(points))
    feature.append(get_feature_13(points))
    return feature

def get_features(input_file):
    fh = open(input_file, 'rt')
    features = []
    classes = []
    avg_points = 0
    cnt = 0
    while True:
        try:
            line = fh.readline()
            line = line.replace('\n','')
            line_split = line.split(';')
            line_len = len(line_split)
            if line_len <= 1:
                break
            avg_points += line_len - 1
            output_class = int(line_split[line_len - 1])
            points = []
            for i in range(line_len - 1):
                point = line_split[i]
                point_split = point.split(',')
                my_point = []
                for p in point_split:
                    my_point.append(float(p))
                points.append(my_point)

            feature = make_features(points)

            features.append(feature)
            classes.append(output_class)
            cnt += 1
        except StopIteration:
            break
    avg_points /= cnt
    print 'Average points in a stroke is ', avg_points
    return (features, classes)

def shuffle(Xs, Ys):
    combined = []
    i_len = len(Xs)
    for i in range(i_len):
        combined.append((Xs[i], Ys[i]))
    random.shuffle(combined)
    Xs = []
    Ys = []
    for (X, Y) in combined:
        Xs.append(X)
        Ys.append(Y)
    return (Xs, Ys)

def normalize(features):
    return preprocessing.normalize(features)

def standardize(features):
    return preprocessing.scale(features)

input_file = '/Users/ankurgupta/Desktop/AI-Project-Kaggle/Rubine/train.csv'
(Xs, Ys) = get_features(input_file)

(Xs, Ys) = shuffle(Xs, Ys)
# Xs = standardize(normalize(Xs))
# print 'After standardizing and normalizing', Xs[2]

# model = LogisticRegression()
# model = naive_bayes.MultinomialNB()
# model = DecisionTreeClassifier()
# model = neighbors.KNeighborsClassifier()
# model = svm.SVC()
model = ensemble.RandomForestClassifier(n_estimators=100)
# model = GradientBoostingClassifier()
# model = AdaBoostClassifier()

split = 250
(XsTrain, YsTrain) = (Xs[:split], Ys[:split])
model.fit(XsTrain, YsTrain)
(XsTest, YsTest) = (Xs[split:], Ys[split:])
predicted = model.predict(XsTest)
cm = metrics.confusion_matrix(YsTest, predicted)
print 'Accuracy score ' , metrics.accuracy_score(YsTest, predicted)
print 'F1 score ', metrics.f1_score(YsTest, predicted, average='micro')
print(cm)

num_folds = 8.0
total_fscore_avg = 0.0
total_accuracy = 0.0

Xs = Xs[:200]
Ys = Ys[:200]
num_times = 5
for j in range(num_times):
    avg_fscore = 0.0
    avg_accuracy = 0.0
    for i in range(int(num_folds)):
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Xs, Ys, test_size=1/num_folds, random_state=i*SEED)
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        f_score = metrics.f1_score(Y_test, predicted, average='micro')
        accuracy = metrics.accuracy_score(Y_test, predicted)
        avg_fscore += f_score
        avg_accuracy += accuracy
    avg_fscore /= num_folds
    avg_accuracy /= num_folds
    total_fscore_avg += avg_fscore
    total_accuracy += avg_accuracy
total_fscore_avg /= (num_times*1.0)
total_accuracy /= (num_times*1.0)

print 'Average f_score is ', total_fscore_avg
print 'Average accuracy is ', total_accuracy


