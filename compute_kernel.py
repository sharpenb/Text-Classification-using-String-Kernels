import numpy as np
import Kernel_implementation.ssk_c2 as ssk
import bs4
from itertools import combinations
import json
from pathlib import Path

import signal
import sys

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

def compute_kernels(bodies, doc_ids, kernels, n, m_lambda):

    enum = list(range(len(bodies)))

    indices1 = zip(enum, enum)
    indices2 = combinations(enum, 2)

    indices = list(indices1) + list(indices2)

    progress_count = 0
    
    # computes K(s, t)
    for i, j in indices:
        s = bodies[i]
        t = bodies[j]
        s_id = doc_ids[i]
        t_id = doc_ids[j]

        key = str(s_id) + " " + str(t_id)
        key2 = str(t_id) + " " + str(s_id)

        if key not in kernels:
            k = ssk.ssk(s, t, n, m_lambda)

            if type(k) is np.ndarray:
                kernels[key] = np.ndarray.tolist(k)
                kernels[key2] = np.ndarray.tolist(k)
            else:
                kernels[key] = k
                kernels[key2] = k
                
        else:
            print("Key exists", key)
        progress_count += 1
        print("Computed", progress_count, "kernels out of", len(indices))
        
    return kernels

# On closing manually, save kernels to file
def signal_handler(signal, frame, data_file, kernels):
        print('You pressed Ctrl+C!')
        print('Saving computed kernels to file')
        with open(data_file, 'w') as f:
            json.dump(kernels, f)
        sys.exit(0)
        


# Open and read file
filename = 'data/train.sgm'
f_train = open(filename)
soup = bs4.BeautifulSoup(f_train.read(), 'html.parser')
articles = soup.find_all('text')

categories = ['corn', 'earn', 'acq', 'crude']

bodies, labels, doc_ids = read_data(articles, "earn")

N = 150
m_lambda =  
n = 5       # Substring length of K

stored_kernels = {}
data_file = "./data/computed_kernels.json"
my_file = Path(data_file)
if my_file.is_file():
    with open(data_file) as f:
        stored_kernels = json.load(f)

# Listen for keyboard interrupts, so we can save our progress so far
signal.signal(signal.SIGINT, lambda signal, frame :  signal_handler(signal, frame, data_file, stored_kernels))

compute_kernels(bodies[0:N], doc_ids[0:N], stored_kernels, n, m_lambda)

with open(data_file, 'w') as f:
    json.dump(stored_kernels, f)

