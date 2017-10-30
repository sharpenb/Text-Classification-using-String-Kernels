import numpy as np
from sklearn.svm import SVC
import Kernel_implementation.ssk as ssk
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

bodies = bodies[0:100]
labels = labels[0:100]



# Divide into train and test
train_test_separator = math.ceil(len(bodies) / 2)
train_bodies = bodies[0:train_test_separator]
train_labels = labels[0:train_test_separator]
test_bodies = bodies[train_test_separator:-1]
test_labels = labels[train_test_separator:-1]


# Initialize SVM
n = 1
m_lambda = 0.5
kernel = lambda x, y : ssk.ssk(x, y, n, m_lambda)
ssvm = StringSVM(kernel)


# Train SVM
start = time.time()

#ssvm.recursive_fit(train_bodies, train_labels, 3)
ssvm.fit(train_bodies, train_labels)

print("Training elapsed time: {:.2f} seconds".format(time.time() - start))

# Test 
start = time.time()

pred = ssvm.predict(test_bodies, test_labels)

print("Prediction elapsed time: {:.2f} seconds".format(time.time() - start))

true_pos = 0
all_pos = 0

for i in range(len(pred)):
    if test_labels[i] == 1:
        all_pos += 1
    if pred[i] == 1:
        true_pos += 1

#print("Precision:", true_pos / len(pred), ", Recall: ", true_pos / all_pos)

