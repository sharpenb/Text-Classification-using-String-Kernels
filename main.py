import numpy as np
from sklearn.svm import SVC
import Kernel_implementation.ssk as ssk
import bs4
import time
from itertools import permutations
import random
import string

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

bodies, labels = read_data(articles, categories[0])
bodies = bodies[0:8]
labels = labels[0:8]

# # Genereate test cases
# bodies = [''.join(p) for p in permutations('abcd')]
# labels = [1 for i in range(len(bodies))]
# bodies2 = [''.join(p) for p in permutations('efgh')]
# labels2 = [0 for i in range(len(bodies2))]
# bodies = bodies + bodies2
# labels = labels + labels2

shuf_bodies = []
shuf_labels = []
index_shuf = list(range(len(bodies)))
random.shuffle(index_shuf)
for i in index_shuf:
    shuf_bodies.append(bodies[i])
    shuf_labels.append(labels[i])

train_test_separator = 4
train_bodies = shuf_bodies[0:train_test_separator]
train_labels = shuf_labels[0:train_test_separator]
test_bodies = shuf_bodies[train_test_separator:-1]
test_labels = shuf_labels[train_test_separator:-1]

X = np.arange(len(shuf_bodies)).reshape(-1, 1)
y = np.array(labels)

X_train = X[0:train_test_separator]
X_test = X[train_test_separator:-1]

y_train = y[0:train_test_separator]
y_test = y[train_test_separator:-1]

start = time.time()

def string_kernel(X, Y, n, m_lambda):
    R = np.zeros((len(X), len(Y)))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            k = int(x[0])
            l = int(y[0])
            
            # SSK Kernel
            R[i, j] = ssk.ssk(shuf_bodies[k], shuf_bodies[l], n, m_lambda)
            # print(shuf_bodies[k], shuf_bodies[l], R[i, j])
            # print(time.time() - start)
    return R

# Fit SVM
clf = SVC(C=100, kernel=lambda x, y : string_kernel(x, y, 1, 0.5))
clf.fit(X_train, y_train) 

print("Training elapsed time: {:.2f} seconds".format(time.time() - start))
print("PREDICTING")

start = time.time()
pred = clf.predict(X_test)
dec = clf.decision_function(X_test)
print("Prediction elapsed time: {:.2f} seconds".format(time.time() - start))

for i in range(len(pred)):
    print(test_bodies[i], pred[i], dec[i])

# print(ssk.ssk('abc', 'abc', 1, 1))
