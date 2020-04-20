import  os
from nltk.stem.porter import PorterStemmer
import re
from collections import defaultdict
import numpy as np


def gather_20newsgroups_data():
    path = '../20news-bydate/'
    dirs = [path + dir_name + '/' for dir_name in os.listdir(path) if not os.path.isfile(path + dir_name)]
    if 'train' in dirs[0]:
        train_dir ,test_dir = (dirs[0],dirs[1])
    else :
        train_dir , test_dir = (dirs[1],dirs[0])
    list_newgroups = [newsgroups for newsgroups in os.listdir(train_dir)]
    list_newgroups.sort()
    #print(train_dir,test_dir)
    return list_newgroups,train_dir,test_dir
def collect_data_from(parent_dir,newsgroups_list):
    data = []
    with open('../20news-bydate/stop_words.txt',"r") as f:
        stop_words = f.read().splitlines()

    stemmer = PorterStemmer()

    for group_id, newsgroup in enumerate(newsgroups_list):
        label = group_id
        dir_path = parent_dir  + newsgroup + '/'
        files = [(filename,dir_path + filename) for filename in os.listdir(dir_path) if os.path.isfile(dir_path + filename)]
        files.sort()
        #print(files)
    for filename,filepath in files:
        with open(filepath,"r") as f:
            text = f.read().upper()
        #remove stop words then stem remaining words
        words = [stemmer.stem(word) for word in re.split('\W+',text) if word not in stop_words]
        #print(words)
        #combine remaining words
        content = " ".join(words)
        assert len(content.splitlines()) == 1
        data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        #print(data)
    return data
def compute_idf(df,corpus_size):
    assert df > 0
    return np.log10(corpus_size * 1./df)
def generate_vocabulary(data_path):
    with open(data_path,"r") as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)

    corpus_size = len(lines)

    for line in lines:
        features = line.split("<fff>")
        text  = features[-1]

        words = list(set(text.split()))

        for word in words:
            doc_count[word]+=1

        words_idfs = [(word,compute_idf(document_freq,corpus_size)) for word,document_freq in zip(doc_count.keys(),doc_count.values()) if document_freq>10 and not word.isdigit()]

        words_idfs.sort(key = lambda abc:abc[1],reverse = True)

        #print(f"Vocabulary size:{len(words_idfs)}")

        with open('../20news-bydate/words_idfs.txt', 'w') as f:
            f.write("\n".join([words + "<fff>" + str(idf) for words,idf in words_idfs]))
def get_tf_idf(data_path):
    #pre-computed idf values
    with open('../20news-bydate/words_idfs.txt', 'r') as f:
        words_idfs = [(line.split("<fff>")[0],float(line.split("<fff>")[1])) for line in f.read().splitlines()]

        word_IDs = dict([(word,index) for index , (word, idf) in enumerate(words_idfs)])

        idfs = dict(words_idfs)

        with open(data_path) as f:
            documents = [(int(line.split("<fff>")[0]),int(line.split("<fff>")[1]),line.split("<fff>")[2]) for line in f.read().splitlines()]

        data_tf_idf = []
        for document in documents:
            label,doc_id,text = document
            words  = [word for word in text.split() if word in idfs]

            print(words)

            word_set = list(set(words))

            max_term_freq = max([words.count(word) for word in word_set])

            words_tfidfs = []

            sum_squares = 0.0

            for word in word_set:
                term_freq = words.count(word)
                tf_idf_value = term_freq * 1./ max_term_freq * idfs[word]
                words_tfidfs.append((word_IDs[word],tf_idf_value))
                sum_squares += tf_idf_value**2

            words_tfidfs_normalized = [str(index) + ":" + str(tf_idf_value/np.sqrt(sum_squares)) for index,tf_idf_value in words_tfidfs]

            sparse_rep = ' '.join(words_tfidfs_normalized)

            data_tf_idf.append((label,doc_id,sparse_rep))

            with open('../20news-bydate/data_tf_idf.txt', 'w') as f:
                f.write("\n".join([str(label) + "<fff>" + str(doc_id) + "<fff>" + str(sparse_rep) for label,doc_id,sparse_rep in data_tf_idf]))

list_newsgroups,train_dir,test_dir = gather_20newsgroups_data()

#print(list_newsgroups)
#train_data = collect_data_from(train_dir,list_newsgroups)

#test_data  = collect_data_from(test_dir,list_newsgroups)

#generate_vocabulary('../20news-bydate/20news-train-processed.txt')

get_tf_idf('../20news-bydate/20news-train-processed.txt')

