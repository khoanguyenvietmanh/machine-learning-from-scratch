import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
class Members:
    def __init__(self,r_d,label = None,doc_id = None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id
class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
    def reset_member(self):
        self._members = []
    def add_member(self,member):
        self._members.append(member)
class Kmeans:
    def __init__(self,num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._E = []
        self._S = []
    def load_data(self,data_path):                        #load data from data_tf_idf file
        def sparse_to_dense(sparse_r_d,vocab_size):       #read tf-idf of each word in 1 file and store in r_d array
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:            #each term consists 2 parts:id and tf-idf value
                index = int(index_tfidf.split(":")[0])    #get the index part
                tfidf = float(index_tfidf.split(":")[1])  #get the tf-idf part
                r_d[index] = tfidf                        #store tf-idf at its index position
            return np.array(r_d)

        with open(data_path,"r") as f:
            d_lines = f.read().splitlines()               # each line (corresponding to 1 text) in file consists of 3 parts:label,doc_id and tf-idf term
        with open("../20news-bydate/words_idfs.txt","r") as f:
            vocab_size = len(f.read().splitlines())        #get total number of vocabs in our corpus
        self._data = []
        self._label_count = defaultdict(int)
        for data_id,d in enumerate(d_lines):
            features  = d.split("<fff>")
            label,doc_id = int(features[0]) , int(features[1])
            self._label_count[label]+=1
            r_d = sparse_to_dense(sparse_r_d = features[2],vocab_size = vocab_size)   #processing the tf-idf term
            self._data.append(Members(r_d,label = label,doc_id= doc_id))      #add member to _data
    def random_init(self,seed_value):                                         #randomly init value for centroid of each cluster
        if seed_value is not None:
            np.random.seed(seed_value)
        initial_centroids = np.random.permutation(len(self._data))[:self._num_clusters]
        for i in range(self._num_clusters):
            self._clusters[i]._centroid = self._data[initial_centroids[i]]._r_d
    def compute_similarity(self,member,centroid):                             #compute similarity between 1 member and 1 centroid
        # here using the cosine similarity between 2 vectors:
        dot_product = np.sum(member * centroid)
        norm_product = np.linalg.norm(member,2)*np.linalg.norm(centroid,2)

        cosine_similarity =  dot_product * 1./norm_product

        return cosine_similarity
    def select_cluster_for(self,member):                                      #select new cluster that fits better than the previous cluster
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member._r_d,cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity
    def update_centroid_of(self,cluster):                                    # after get all of members that belong to that cluster,update the centroid of that one
        member_r_ds = np.array([member._r_d for member in cluster._members],dtype = np.float16)
        aver_r_d = np.mean(member_r_ds,axis = 0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d**2))
        new_centroid = np.array([value/sqrt_sum_sqr for value in aver_r_d])

        cluster._centroid = new_centroid

    def stopping_condition(self,criterion,threshold):                        # choosing which conditions want to stop the training process
        criteria = ['centroid','similarity','max_iters']
        assert criterion in criteria
        if criterion == "max_iters":
            if self._iteration >= threshold:                                 # if the number of iterations reach the maximum iteration value
                return True
            else:
                return False
        elif criterion == "centroid":
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]

            self._E = E_new
            if(len(E_new_minus_E) <= threshold):                            # if the list of cluster centroid does not change significantly(E_new/E_old <= threshold)
                return True
            else:
                return False
        else:
            new_S_minus_S = self._new_S - self._S                           # The loss error does not change much(or S_new - S_old <= threshold)
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False
    def run(self,seed_value,criterion,threshold):                           #  run the training process
        self.random_init(seed_value)

        self._iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_member()
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)

            self._iteration+=1
            if self.stopping_condition(criterion,threshold):
                break
    def compute_purity(self):                                    # compute the purity after clustering
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum+=max_count
        return majority_sum * 1./len(self._data)
    def compute_NMI(self):                                         # compute the normalized mutual information
        I_value,H_omega,H_C,N = 0.,0.,0.,len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omega += -wk/(N * np.log10(wk / N))
            member_labels = [member._label for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += (wk_cj / N) * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. /(H_omega + H_C)

def load_data(data_path):                             #load data from data_tf_idf file
    def sparse_to_dense(sparse_r_d,vocab_size):       #read tf-idf of each word in 1 file and store in r_d array
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:            #each term consists 2 parts:id and tf-idf value
            index = int(index_tfidf.split(":")[0])    #get the index part
            tfidf = float(index_tfidf.split(":")[1])  #get the tf-idf part
            r_d[index] = tfidf                        #store tf-idf at its index position
        return np.array(r_d,dtype=np.float16)

    with open(data_path,"r") as f:
        d_lines = f.read().splitlines()               # each line (corresponding to 1 text) in file consists of 3 parts:label,doc_id and tf-idf term
    with open("../20news-bydate/words_idfs.txt","r") as f:
        vocab_size = len(f.read().splitlines())        #get total number of vocabs in our corpus
    _data = []
    _label = []
    _label_count = defaultdict(int)
    for data_id,d in enumerate(d_lines):
        features  = d.split("<fff>")
        label,doc_id = int(features[0]) , int(features[1])
        _label.append(label)
        _label_count[label]+=1
        r_d = sparse_to_dense(sparse_r_d = features[2],vocab_size = vocab_size)   #processing the tf-idf term
        _data.append(r_d)
    return _data,_label
def clustering_with_Kmeans():
    data,labels = load_data("../20news-bydate/data_tf_idf.txt")
    print(labels)
    X = csr_matrix(data)
    print("========")
    kmeans = KMeans(n_clusters=20,init = "random",n_init=5,tol = 1e-3,random_state=2019).fit(X)
    labels = np.array(kmeans.labels_)
    print(labels)
def compute_accuracy(predicted_y,expected_y):
    matches = np.equal(predicted_y,expected_y)                                          # check the predicted set and the test set is the same of not by element-wise
    accuracy = np.sum(matches).astype(float)/len(predicted_y)                           # accuracy = total number of predicted value matches together/number of predicted values

    return accuracy
def classifying_with_linear_SVMs():
    train_X,train_y = load_data("../20news-bydate/data_tf_idf.txt")
    classifier = LinearSVC(C = 10.0,tol = 0.001,verbose = True)                              # using the Soft SVM
    #classifier = SVC(C = 50.0,kernel = "rbf",gamma = 0.1,tol = 0.001,verbose = True)        # using the Kernel SVM
    classifier.fit(train_X,train_y)

    test_X,test_y = load_data("../20news-bydate/test_data_tf_idf.txt")
    predict_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predict_y,expected_y= test_y)

    print(f"\nAccuracy of SVM is : {accuracy}")
if __name__ == "__main__":
    km = Kmeans(20)
    km.load_data("../20news-bydate/data_tf_idf.txt")
    km.run(seed_value = 3,criterion="centroid",threshold = 10)
    print(f"Purity:{km.compute_purity()}")
    print(f"NMI:{km.compute_NMI()}")

    #classifying_with_linear_SVMs()
    #clustering_with_Kmeans()


