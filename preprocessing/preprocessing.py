import os
import bs4
import string


def remove_stop_words(doc, stopwords):
    new_doc = ' '.join([word for word in doc.split() if word not in stopwords]) 
    return new_doc

def remove_punctuations(doc):
    punc = string.punctuation
    new_doc = ''.join([word for word in list(doc) if word not in punc]) 
    return new_doc

def save_docs(docs, labels, doc_ids, filename):
    f = open(filename, 'w')

    for d in docs:
        lab = labels[d]
        doc_id = doc_ids[d]
        
        f.write('<TEXT ID=' + str(doc_id) + '>\n')

        f.write('<TOPICS>\n')
        for l in lab:
            f.write('<D>' + l + '</D>')
        f.write('\n</TOPICS>\n')

        f.write('<BODY>\n')
        f.write(d)
        f.write('</BODY>\n')
        f.write('</TEXT>\n')
        
    


# get stop words
f = open("../data/stop-word-list.txt", 'r')
stopwords = f.read().split()

# directory of data files
data_dir = "../data/reuters21578/"

# read documents 
docs = []
for filename in os.listdir(data_dir):
    if filename.endswith(".sgm"):
        f = open(data_dir + filename, 'r')
        docs.append(f.read())

# label data and
# split documents into training and testing
labels = {}
doc_ids = {}
train_articles = []
test_articles = []

id_count = 0
for doc in docs:
    soup = bs4.BeautifulSoup(doc, 'html.parser')

    articles = soup.find_all('reuters')

    for a in articles:

        body = a.find('body')
        if body != None:
            text = body.string

            # check if in train or test set
            topics = a.get('topics')
            lewissplit = a.get('lewissplit')

            if(topics == 'YES' and lewissplit == 'TRAIN'):
                text_new = remove_stop_words(text, stopwords)            
                text_new = remove_punctuations(text_new)
                train_articles.append(text_new)
            elif(topics == 'YES' and lewissplit == 'TEST'):
                text_new = remove_stop_words(text, stopwords)            
                text_new = remove_punctuations(text_new)
                test_articles.append(text_new)
            else:
                continue


            # extract categories
            categories = []
            t = a.find('topics')
            for c in t.find_all('d'):
                categories.append(c.string)

            labels[text_new] = categories

            doc_ids[text_new] = id_count
            id_count += 1

save_docs(train_articles, labels, doc_ids, '../data/train.sgm')
save_docs(test_articles, labels, doc_ids, '../data/test.sgm')
