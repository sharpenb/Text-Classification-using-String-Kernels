import numpy as np
from sklearn.svm import SVC
import Kernel_implementation.ssk_c as ssk
from svm.string_svm import StringSVM
import bs4
import time
from itertools import permutations
import random
import string
import math


def read_data(articles, category):
    labels = []
    bodies = []

    for a in articles:

        topics = a.find('topics')
        found_category = False
        for c in topics.find_all('d'):
            if c.string == category:
                labels.append(1)
                found_category = True
        if not found_category:
            labels.append(0)

        body = a.find("body")
        bodies.append(body.string)

    return bodies, labels


# Open and read file
filename = 'data/train.sgm'
f_train = open(filename)
soup = bs4.BeautifulSoup(f_train.read(), 'html.parser')
articles = soup.find_all('text')

categories = ['corn', 'earn', 'acq', 'crude']

bodies, labels = read_data(articles, categories[1])

bodies = bodies[0:200]
labels = labels[0:200]


# Divide into train and test
train_test_separator = 150
train_bodies = bodies[0:train_test_separator]
train_labels = labels[0:train_test_separator]
test_bodies = bodies[train_test_separator:-1]
test_labels = labels[train_test_separator:-1]


# Initialize SVM
n_vals = [1, 2, 3]

precisions = []
recalls = []

for n in n_vals:
    m_lambda = 0.9
    kernel = lambda x, y : ssk.ssk(x, y, n, m_lambda)
    ssvm = StringSVM(kernel)


    # Train SVM
    start = time.time()

    ssvm.recursive_fit(train_bodies, train_labels, 3)
    #ssvm.fit(train_bodies, train_labels)

    print("Training elapsed time: {:.2f} seconds".format(time.time() - start))

    # Test 
    start = time.time()

    pred = ssvm.predict(test_bodies, test_labels)

    print("Prediction elapsed time: {:.2f} seconds".format(time.time() - start))

    true_pos = 0
    all_pos = 0
    all_pred_pos = 0

    for i in range(len(pred)):
        if test_labels[i] == 1:
            all_pos += 1
        if pred[i] == 1 and test_labels[i] == 1:
            true_pos += 1
        if pred[i] == 1:
            all_pred_pos += 1

    recall = true_pos / all_pos
    precision = true_pos / all_pred_pos
    
    recalls.append(recall)
    precisions.append(precision)

    print(true_pos, all_pos)

    print("Precision:", precision, ", Recall: ", recall)

f = open('output_c.txt', 'w')

f.write('precision:\n')
f.write(str(precisions))
f.write("\nrecall:\n")
f.write(str(recalls))