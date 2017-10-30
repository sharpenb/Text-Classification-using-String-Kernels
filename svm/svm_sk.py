import numpy as np
from sklearn.svm import SVC



data = ['ahed','bdeded','adeded','bdede']
X = np.arange(len(data)).reshape(-1, 1)
#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

def string_kernel(X, Y):
        R = np.zeros((len(x), len(y)))
        for x in X:
            for y in Y:
                i = int(x[0])
                j = int(y[0])
                # simplest kernel ever
                R[i, j] = data[i][0] == data[j][0]
        return R


clf = SVC()
clf.fit(X, y) 
SVC(C=1.0, kernel=string_kernel)

print(clf.predict())

