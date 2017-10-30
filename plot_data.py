import numpy as np
import matplotlib.pyplot as plt
from eval_ssk import eval_ssk
import json
import bs4

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
filename = './data/train.sgm'
f_train = open(filename)
soup = bs4.BeautifulSoup(f_train.read(), 'html.parser')
articles = soup.find_all('text')

bodies, labels, doc_ids = read_data(articles, "earn")

N = 150
doc_ids = doc_ids[0:N]
labels = labels[0:N]

# divide data for cross val
points_per_group = 15

subseq_lengths = [1, 2, 3, 4, 5]
decay_factors = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decay_factors_str = ['1e-2', '1e-1', '2e-1', '3e-1', '4e-1', '5e-1', '6e-1', '7e-1', '8e-1', '9e-1']

f1_mean = np.zeros((len(subseq_lengths), len(decay_factors)))
f1_std = np.zeros((len(subseq_lengths), len(decay_factors)))
p_mean = np.zeros((len(subseq_lengths), len(decay_factors)))
p_std = np.zeros((len(subseq_lengths), len(decay_factors)))
r_mean = np.zeros((len(subseq_lengths), len(decay_factors))) 
r_std = np.zeros((len(subseq_lengths), len(decay_factors)))


for l in range(len(decay_factors)):
    # load kernel
    data_file = './data/computed_kernels_lambda' + decay_factors_str[l] + '.json'
    with open(data_file) as f:
        kernel = json.load(f)

    for n in range(len(subseq_lengths)):
        res_mean, res_std = eval_ssk(n + 1, kernel, doc_ids, labels, points_per_group) 
        
        print()
        print(decay_factors[l], n)
        print(res_mean)

        f1_mean[n, l] = res_mean["f1"]
        f1_std[n, l] = res_std["f1"]

        p_mean[n, l] = res_mean["precision"]
        p_std[n, l] = res_std["precision"]

        r_mean[n, l] = res_mean["recall"]
        r_std[n, l] = res_std["recall"]

# Values from Lodhi et al

decay_factors_article = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]
f1_mean_article = [0.946, 0.946, 0.944, 0.944, 0.944, 0.944, 0.943, 0.936, 0.928, 0.914]
p_mean_article = [0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.994, 0.989]
r_mean_article = [0.905, 0.905, 0.903, 0.903, 0.902, 0.903, 0.900, 0.888, 0.873, 0.853]



for n in range(len(subseq_lengths)):
    plt.figure(1)
    ax = plt.gca()
    plt.plot(decay_factors, f1_mean[n, :], lw = 2, label = r'$n = ' + str(n + 1)+'$')
    
    plt.figure(2)
    ax = plt.gca()
    plt.plot(decay_factors, p_mean[n, :], lw = 2, label = r'$n = ' + str(n + 1) + '$')
    
    plt.figure(3)
    ax = plt.gca()
    plt.plot(decay_factors, r_mean[n, :], lw = 2, label = r'$n = ' + str(n + 1)+'$')

plt.figure(1)

plt.plot(decay_factors_article, f1_mean_article, lw = 2, ls='--', color='m', label = r'$n = 5$, Lodhi et al.')

ax = plt.gca()
ax.set_ylim([0.6, 1.05])
ax.set_xlim([0, decay_factors[-1]+0.1])
plt.legend(ncol=2)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel(r'$F_1$',fontsize=16)
plt.savefig('./data/results/f1.pdf', format='pdf', dpi=300)


plt.figure(2)

plt.plot(decay_factors_article, p_mean_article, lw = 2, ls='--', color='m', label = r'$n = 5$, Lodhi et al.')

ax = plt.gca()
ax.set_ylim([0.6, 1.05])
ax.set_xlim([0, decay_factors[-1]+0.1])
plt.legend(loc=4, ncol=2)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel(r'$p$',fontsize=16)
plt.savefig('./data/results/recall.pdf', format='pdf', dpi=300)


plt.figure(3)

plt.plot(decay_factors_article, r_mean_article, lw = 2, ls='--', color='m', label = r'$n = 5$, Lodhi et al.')

ax = plt.gca()
ax.set_ylim([0.6, 1.05])
ax.set_xlim([0, decay_factors[-1]+0.1])
plt.legend(ncol=2)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel(r'$r$',fontsize=16)
plt.savefig('./data/results/precision.pdf', format='pdf', dpi=300)


plt.show()
