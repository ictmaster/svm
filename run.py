#!/usr/bin/python
import time
import sys
import numpy as np
import random
from sklearn import svm

import warnings
warnings.simplefilter("ignore")

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
    train_dict = {}
    # Populating the dictionary with training data
    for row in training:
        if row[-1] != '':
            l = train_dict.get(row[-1], [])
            l.append(row[:-1])
            train_dict[row[-1]] = l
    xlist = []
    ylist = []
    for x in train_dict.iteritems():
        # List of lists
        xlist.extend(x[1])
        ylist.extend([x[0] for i in x[1]])

    X = np.array(xlist)
    Y = np.array(ylist)

    C = 1.0
    gamma = 0.5

    svm_linear = svm.SVC(kernel='linear',C=C,gamma=gamma).fit(X,Y)

    import pdb;pdb.set_trace()



    print "Script executed in {0} seconds...".format("%.2f"%(time.time()-script_start_time))
