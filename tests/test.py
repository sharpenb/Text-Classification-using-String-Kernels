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

####################
#                  #
# SIMPLE TEST CASE #
#                  #
####################
print("####### SIMPLE TEST CASE ########")

# Generate data
bodies = [''.join(p) for p in permutations('abcd')]
labels = [1 for i in range(len(bodies))]
bodies2 = [''.join(p) for p in permutations('edfg')]
labels2 = [0 for i in range(len(bodies2))]
bodies = bodies + bodies2
labels = labels + labels2

# Shuffle
shuf_bodies = []
shuf_labels = []
index_shuf = list(range(len(bodies)))
random.shuffle(index_shuf)
for i in index_shuf:
    shuf_bodies.append(bodies[i])
    shuf_labels.append(labels[i])

# Divide into train and test
train_test_separator = math.ceil(len(bodies) / 2)
train_bodies = shuf_bodies[0:train_test_separator]
train_labels = shuf_labels[0:train_test_separator]
test_bodies = shuf_bodies[train_test_separator:-1]
test_labels = shuf_labels[train_test_separator:-1]


# Initialize SVM
n = 1
m_lambda = 0.5
kernel = lambda x, y : ssk.ssk(x, y, n, m_lambda)
ssvm = StringSVM(kernel)

# Train SVM
start = time.time()

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

print("Precision:", true_pos / len(pred), ", Recall: ", true_pos / all_pos)

