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

subseq_lengths = np.arange(1,15)
decay_factors = [0.6]
decay_factors_str = ['6e-1_n14']

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
        print(decay_factors[0], n)
        print(res_mean)

        f1_mean[n, l] = res_mean["f1"]
        f1_std[n, l] = res_std["f1"]

        p_mean[n, l] = res_mean["precision"]
        p_std[n, l] = res_std["precision"]

        r_mean[n, l] = res_mean["recall"]
        r_std[n, l] = res_std["recall"]

# Values from Lodhi et al
subseq_lengths_article = [3,4,5,6,7,8,10,12,14]
f1_mean_article = [0.925, 0.932, 0.936, 0.936, 0.940, 0.934, 0.927, 0.931, 0.936]
p_mean_article = [0.981, 0.992, 0.992, 0.992, 0.992, 0.992, 0.997, 0.981, 0.959]
r_mean_article = [0.878, 0.888, 0.888, 0.888, 0.900, 0.885, 0.868, 0.888, 0.915]

ax = plt.gca()
f1_plot = plt.plot(subseq_lengths, f1_mean[:, 0], lw = 2, label = r'$F_1$')
p_plot = plt.plot(subseq_lengths, p_mean[:, 0], lw = 2, label = r'$p$')
r_plot = plt.plot(subseq_lengths, r_mean[:, 0], lw = 2, label = r'$r$')


plt.plot(subseq_lengths_article, f1_mean_article, color = f1_plot[0].get_color(), ls = '--', lw = 2, label = r'$F_1$ Lodhi et al')
plt.plot(subseq_lengths_article, p_mean_article, color = p_plot[0].get_color(), ls = '--', lw = 2, label = r'$p$ Lodhi et al')
plt.plot(subseq_lengths_article, r_mean_article, color = r_plot[0].get_color(), ls = '--', lw = 2, label = r'$r$ Lodhi et al')


ax = plt.gca()
ax.set_ylim([0, 1.05])
ax.set_xlim([0, subseq_lengths[-1]+1])
plt.legend(ncol=2, loc=4)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$n$',fontsize=16)
#plt.ylabel(r'$$',fontsize=16)
plt.savefig('./data/results/results_n14.pdf', format='pdf', dpi=300)

plt.show()
