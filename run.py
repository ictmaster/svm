#!/usr/bin/python
import time
import sys
import numpy as np
import random
from sklearn import svm

import warnings
warnings.simplefilter("ignore")

# Function for testing svm
def test_svm(svm, testing_dict):
    num_correct = 0
    num_wrong = 0
    for correct, testing in testing_dict.iteritems():
        for test in testing:
            r = svm.predict(test)[0]
            if r == correct:
                num_correct += 1
            else:
                num_wrong += 1

    print("\tCorrect:{0}".format(num_correct))
    print("\tWrong:{0}".format(num_wrong))
    accuracy = float(num_correct)/(num_correct+num_wrong)*100
    print("\tAccuracy:{0:.2f}%".format(round(accuracy,2)))

# Populating the dictionary with training data
def generate_data_dict(data):
    d = {}
    for row in data:
        if row[-1] != '':
            l = d.get(row[-1], [])
            l.append(row[:-1])
            d[row[-1]] = l
    return d

if __name__ == '__main__':
    script_start_time = time.time()

    # Load all data from a file
    all_data = [i.rstrip().split(',') for i in open('iris.data').readlines()]
    random.shuffle(all_data)
    training_ratio = 0.7
    num_training = int(len(all_data)*training_ratio)
    training = all_data[:num_training]
    testing = all_data[num_training:]

    # A dictionary to store all data
    train_dict = generate_data_dict(training)
    test_dict = generate_data_dict(testing)

    xlist = []
    ylist = []
    for x in train_dict.iteritems():
        # List of lists
        xlist.extend(x[1])
        ylist.extend([x[0] for i in x[1]])

    X = np.array([[float(y) for y in x] for x in xlist])
    Y = np.array(ylist)
    C = 1.0
    gamma = 0.5

    print("Training Linear svm")
    svm_linear = svm.SVC(kernel='linear',C=C,gamma=gamma).fit(X,Y)
    print("Training Polynomial svm")
    svm_polynomial = svm.SVC(kernel='poly',C=C,gamma=gamma).fit(X,Y)
    print("Training RBF svm")
    svm_rbf = svm.SVC(kernel='rbf',C=C,gamma=gamma).fit(X,Y)
    print("Training Sigmoid svm")
    svm_sigmoid = svm.SVC(kernel='sigmoid',C=C,gamma=gamma).fit(X,Y)
    h = 0.2 #Mesh step
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    print("\nLinear Test:")
    test_svm(svm_linear,test_dict)
    print("\nPolynomial Test:")
    test_svm(svm_polynomial,test_dict)
    print("\nRBF Test:")
    test_svm(svm_rbf,test_dict)
    print("\nSigmoid Test:")
    test_svm(svm_sigmoid,test_dict)

    print ("\nScript executed in {0} seconds...".format("%.2f"%(time.time()-script_start_time)))
