import numpy as np
import matplotlib.pyplot as plt
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis = 0)

original_label = np.asarray([0]*N + [1]*N + [2]*N).T
def visualize_data(X,label,cluster_centroid):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.plot(cluster_centroid[:, 0], cluster_centroid[:, 1], "y^",markersize = 10)
    plt.show()
class K_meams_clustering:
    def __init__(self,K,epochs):
        self.K = K
        self.epochs = epochs
    def kmeans_init_center(self):
        cluster_centroid = X[np.random.choice(X0.shape[0],self.K,replace=True)]
        return cluster_centroid
    def kmeans_assign_label(self,X,cluster_centroid):
        label = []
        for i in range(X.shape[0]):
            data = np.array([X[i,:] for j in range(self.K)])
            D = np.sum((data-cluster_centroid)**2,axis=1)
            label.append(D.argmin())
        label = np.array(label)
        return label
    def kmeans_update_centroid(self, X, label, cluster_centroid):
        for i in range(self.K):
            temp = [j for j in range(len(label)) if label[j] == i]
            X_temp = np.array(X[temp])
            cluster_centroid[i] = (X_temp[:,0].mean(),X_temp[:,1].mean())
        return cluster_centroid
    def stopping_criteria(self,old_cluster_centroid,new_cluster_centroid):
        return np.all(old_cluster_centroid == new_cluster_centroid)
    def kmeans(self,X):
        cluster_centroid = self.kmeans_init_center()
        for i in range(self.epochs):
            label = self.kmeans_assign_label(X,cluster_centroid)
            old_cluster_centroid = np.copy(cluster_centroid)
            cluster_centroid = self.kmeans_update_centroid(X,label,cluster_centroid)
            if self.stopping_criteria(old_cluster_centroid,cluster_centroid):
               break
        return cluster_centroid

if __name__ =="__main__":
    km = K_meams_clustering(3,100)
    cluster_centroid = km.kmeans(X)
    print(cluster_centroid)
    visualize_data(X, original_label, cluster_centroid)