#!/usr/bin/python
from __future__ import print_function
import time
import sys
import numpy as np
import random
from sklearn import svm, grid_search
import threading


import warnings
warnings.simplefilter("ignore")

# Function for testing svm
def test_svm(svm, testing_dict, name):
    num_correct = 0
    num_wrong = 0
    for correct, testing in testing_dict.items():
        for test in testing:
            r = svm.predict(test)[0]
            if r == correct:
                num_correct += 1
            else:
                num_wrong += 1
    print("\n{1} - Correct:{0}".format(num_correct, name), end="")
    print("\n{1} - Wrong:{0}".format(num_wrong, name), end="")
    accuracy = float(num_correct)/(num_correct+num_wrong)*100
    print("\n{1} - Accuracy:{0:.2f}%".format(round(accuracy,2), name), end="")

def train_svm(kernel, X,Y, test_dict):
    time_train = time.time()
    time_total = time.time()
    print("\nTraining {0} svm...".format(kernel), end="")
    trained_svm = svm.SVC(kernel=kernel,C=C,gamma=gamma).fit(X,Y)
    print("\nDone training {0}...".format(kernel), end="")
    print("\n{0} training took {1:.2f} seconds...".format(kernel, round(time.time()-time_train,2)), end="")
    print("\nTesting {0}...".format(kernel), end="")
    time_test = time.time()
    test_svm(trained_svm,test_dict, kernel)
    print("\n{0} testing took {1:.2f}".format(kernel, round(time.time()-time_test,2)), end="")


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

    #file_names = ['poker-hand-training-true.data', 'poker-hand-testing.data']
    file_names = ['iris.data']
    print("Loading files...")
    if len(file_names) == 1:
        # Load all data from a file
        all_data = [i.rstrip().split(',') for i in open(file_names[0]).readlines()]
        random.shuffle(all_data)
        training_ratio = 0.7
        num_training = int(len(all_data)*training_ratio)
        training = all_data[:num_training]
        testing = all_data[num_training:]
    elif len(file_names) == 2:
        training = [i.rstrip().split(',') for i in open(file_names[0]).readlines()]
        testing = [i.rstrip().split(',') for i in open(file_names[1]).readlines()]
    else:
        print("Wrong number of files...")
        sys.exit(1)
    # A dictionary to store all data
    print("Generating dictionaries...")
    train_dict = generate_data_dict(training)
    test_dict = generate_data_dict(testing)

    print("Generating arrays...")
    xlist = []
    ylist = []
    for x in train_dict.items():
        # List of lists
        xlist.extend(x[1])
        ylist.extend([x[0] for i in x[1]])

    X = np.array([[float(y) for y in x] for x in xlist])
    Y = np.array(ylist)
    C = 1.0
    gamma = 0.5

    linear = threading.Thread(name='linear', target=train_svm, args=['linear', X, Y, test_dict])
    poly = threading.Thread(name='poly', target=train_svm, args=['poly', X, Y, test_dict])
    rbf = threading.Thread(name='rbf', target=train_svm, args=['rbf', X, Y, test_dict])
    sigmoid = threading.Thread(name='sigmoid', target=train_svm, args=['sigmoid', X, Y, test_dict])

    threads = [linear, poly, rbf, sigmoid]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


    h = 0.2 #Mesh step
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    print ("\nScript executed in {0} seconds...".format("%.2f"%(time.time()-script_start_time)))
