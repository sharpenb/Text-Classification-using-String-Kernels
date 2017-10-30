import numpy as np
import json
from sklearn.svm import SVC
import bs4

def gram_matrix(indices1, indices2, kernel, n): 
    
    R = np.zeros((len(indices1), len(indices2)))

    for i in range(len(indices1)):
        for j in range(len(indices2)):
            k = int(indices1[i])
            l = int(indices2[j])

            v1 = kernel[str(k) + " " + str(l)][n - 1]
            v2 = kernel[str(k) + " " + str(k)][n - 1]
            v3 = kernel[str(l) + " " + str(l)][n - 1]
            
            v = v1 / np.sqrt(v2 * v3)

            R[i, j] = v

    return R

def read_data(articles, category):
    doc_ids = []
    labels = []
    bodies = []

    for a in articles:

        doc_id = int(a.get('id'))
        doc_ids.append(doc_id)

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

    return bodies, labels, doc_ids


# load documents
filename = '../data/train.sgm'
f_train = open(filename)
soup = bs4.BeautifulSoup(f_train.read(), 'html.parser')
articles = soup.find_all('text')

bodies, labels, doc_ids = read_data(articles, "earn")

# load kernel
with open('../data/computed_kernels_lambda3e-1.json') as f:
    kernels = json.load(f)

N = 150
doc_ids = doc_ids[0:N]
labels = labels[0:N]
count = 0
for l in labels:
    if l == 1:
        count += 1

print(count)

# divide data for cross val
points_per_group = 15

all_prec = []
all_recall = []
all_f1 = []

k = 0
while k < N:

    test_indices = np.arange(k, k + points_per_group)
    test_ids = np.take(doc_ids, test_indices)

    # filter out the test data
    training_indices = np.arange(0, k) 
    training_indices = np.append(training_indices, np.arange(k + points_per_group, N))
    training_ids = np.take(doc_ids, training_indices)

    k += points_per_group

    training_labels = np.take(labels, training_indices)
    test_labels = np.take(labels, test_indices)

    K_train = gram_matrix(training_ids, training_ids, kernels, 2)
    K_test = gram_matrix(test_ids, training_ids, kernels, 2)

    svm = SVC(kernel = 'precomputed')
    svm.fit(K_train, training_labels)
    pred = svm.predict(K_test)
    


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
    f1 = 2 * precision * recall / (precision + recall)
    
    all_recall.append(recall)
    all_prec.append(precision)
    all_f1.append(f1)

    print("Precision:", precision, ", Recall: ", recall, ", F1: ", f1)
    

print("MEAN")
print("Precision:", np.mean(all_prec), ", Recall: ", np.mean(all_recall), ", F1: ", np.mean(all_f1))
