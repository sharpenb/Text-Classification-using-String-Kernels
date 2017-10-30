import numpy as np
from sklearn.svm import SVC
import Kernel_implementation.ssk_c2 as ssk
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

# Get the bodies and labels of different categories
bodies_corn, labels_corn = read_data(articles, categories[0])
bodies_earn, labels_earn = read_data(articles, categories[1])

# Get the indices for true corn and earn documents
true_corn_idx =  [i for i, j in enumerate(labels_corn) if j == 1]
true_earn_idx =  [i for i, j in enumerate(labels_earn) if j == 1]

m_lambda = 0.5
n = 5
print("Kernel values for n={:d}, lambda={:.2f}".format(n, m_lambda))
# Corn/Corn
print("Printing for corn/corn kernel values")
print("C0 : C0\t", ssk.ssk(bodies_corn[true_corn_idx[0]], bodies_corn[true_corn_idx[0]], n, m_lambda))
print("C1 : C1\t", ssk.ssk(bodies_corn[true_corn_idx[1]], bodies_corn[true_corn_idx[1]], n, m_lambda))
print("C0 : C1\t", ssk.ssk(bodies_corn[true_corn_idx[0]], bodies_corn[true_corn_idx[1]], n, m_lambda))

# Earn/Earn
print("Printing for corn/corn kernel values")
print("E0 : E0\t", ssk.ssk(bodies_earn[true_earn_idx[0]], bodies_earn[true_earn_idx[0]], n, m_lambda))
print("E1 : E1\t", ssk.ssk(bodies_earn[true_earn_idx[1]], bodies_earn[true_earn_idx[1]], n, m_lambda))
print("E0 : E1\t", ssk.ssk(bodies_earn[true_earn_idx[0]], bodies_earn[true_earn_idx[1]], n, m_lambda))

#Corn/Earn
print("Printing for corn/earn kernel values")
print("C0 : E0\t", ssk.ssk(bodies_corn[true_corn_idx[0]], bodies_earn[true_earn_idx[0]], n, m_lambda))
print("C0 : E1\t", ssk.ssk(bodies_corn[true_corn_idx[0]], bodies_earn[true_earn_idx[1]], n, m_lambda))
print("C1 : E0\t", ssk.ssk(bodies_corn[true_corn_idx[1]], bodies_earn[true_earn_idx[0]], n, m_lambda))
print("C1 : E1\t", ssk.ssk(bodies_corn[true_corn_idx[1]], bodies_earn[true_earn_idx[1]], n, m_lambda))


