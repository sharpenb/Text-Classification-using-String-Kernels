import numpy as np
import json
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
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

def eval_ssk(n, lambda_kernel, doc_ids, labels, points_per_group):
    N = len(doc_ids)

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

        K_train = gram_matrix(training_ids, training_ids, lambda_kernel, n)
        K_test = gram_matrix(test_ids, training_ids, lambda_kernel, n)

        C_range = np.logspace(-2, 10, 100, base = 2)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(kernel='precomputed',class_weight='balanced'), param_grid=param_grid, cv=cv)
        grid.fit(K_train, training_labels)

        print(grid.best_params_)

        svm = SVC(C = grid.best_params_["C"], kernel = 'precomputed', class_weight='balanced')
        #svm = SVC(C=1,kernel = 'precomputed', class_weight='balanced')

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


        if all_pred_pos != 0 and all_pos != 0:
            recall = true_pos / all_pos
            precision = true_pos / all_pred_pos
            f1 = 2 * precision * recall / (precision + recall)
            
            all_recall.append(recall)
            all_prec.append(precision)
            all_f1.append(f1)

        
    res_mean = {}
    res_std = {}
    res_mean["precision"], res_std["precision"] = np.mean(all_prec), np.std(all_prec)
    res_mean["recall"], res_std["recall"] = np.mean(all_recall), np.std(all_recall)
    res_mean["f1"], res_std["f1"] = np.mean(all_f1), np.std(all_f1)

    return res_mean, res_std

