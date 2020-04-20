import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
class Linear_Regression:
    def __init__(self,epochs,learning_rate,LAMBDA):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.LAMBDA = LAMBDA
    def _hypothesis(self,X_Train):
        return np.dot(X_Train,self.w)

    def _calculate_loss_function(self,X_Train,Y_Train):
        m = X_Train.shape[0]
        return (1/(2*m))*np.sum((self._hypothesis(X_Train) - Y_Train)**2) + (self.LAMBDA/(2*m))*np.linalg.norm(self.w[1:],2)**2

    def _gradient(self,X_Train,Y_Train):
        m = X_Train.shape[0]
        return np.dot((self._hypothesis(X_Train) - Y_Train).T,X_Train[:,0]).T/m,np.dot((self._hypothesis(X_Train) - Y_Train).T,X_Train[:,1:]).T/m + (self.LAMBDA/m)*self.w[1:]
    def train(self,X_Train,Y_Train):
        loss_function = []
        for epoch in range(self.epochs):
            w0_grad,w_grad = self._gradient(X_Train,Y_Train)
            self.w[0] = self.w[0] - self.learning_rate * w0_grad
            self.w[1:] = self.w[1:] - self.learning_rate * w_grad
            if np.linalg.norm(w_grad,2) < 1e-3:
                break
            loss_function.append(self._calculate_loss_function(X_Train,Y_Train))
        return loss_function

    def standarnize_training_set(self,X_Train,Y_Train):
        X_mean = X_Train.mean(axis = 0)
        X_std = X_Train.std(axis = 0)
        Y_mean = np.mean(Y_Train)
        Y_std = np.std(Y_Train)

        X_mean = np.array([X_mean for i in range(X_Train.shape[0])])
        X_std = np.array([X_std for i in range(X_Train.shape[0])])

        Y_mean = np.full((Y_Train.shape[0],1),Y_mean)
        Y_std = np.full((Y_Train.shape[0],1),Y_std)

        return (X_Train-X_mean)/X_std,(Y_Train-Y_mean)/Y_std

    def init_parameter(self,X_Train):
        self.w = np.random.randn(X_Train.shape[1],1)

    def predict(self,X_Test):
        assert X_Test.shape[1] == self.w.shape[0], "Incorrect shape."
        return self._hypothesis(X_Test)

    def r2_score(self, y_hat, y_test):
        total_sum_squares = np.sum((y_test - np.mean(y_test))**2)
        residual_sum_squares = np.sum((y_test - y_hat)**2)
        return 1 - residual_sum_squares/total_sum_squares


def main():
    X = np.loadtxt("prostate.data.txt",skiprows=1)
    learning_rate = 0.01
    epochs = 500
    LAMBDA = 0.1
    X_Train = X[:,:-1]
    Y_Train = np.array(X[:,-1]).reshape(X_Train.shape[0],1)
    X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Train,Y_Train,test_size=0.2)
    linear_regression = Linear_Regression(epochs,learning_rate,LAMBDA)

    X_normalized,Y_normalized = linear_regression.standarnize_training_set(X_Train,Y_Train)

    X_Test_normalized,Y_Test_normalized = linear_regression.standarnize_training_set(X_Test,Y_Test)

    ones = np.ones((X_normalized.shape[0],1))

    ones_2 = np.ones((X_Test_normalized.shape[0],1))

    X_normalized = np.concatenate([ones,X_normalized],axis = 1)

    X_Test_normalized = np.concatenate([ones_2,X_Test_normalized],axis = 1)

    linear_regression.init_parameter(X_normalized)

    loss_function = linear_regression.train(X_normalized,Y_normalized)

    '''

    i = range(epochs)

    plt.ylabel("Loss function")

    plt.xlabel("Number of epochs")

    plt.plot(i,loss_function,color = "r")

    plt.show()

    '''
    x = range(0,X_Test_normalized.shape[0]-10,1)
    plt.plot(x,linear_regression.predict(X_Test_normalized[:10,:]),"go")
    plt.plot(x,Y_Test_normalized[:10,:],"ro")

    plt.show()


main()




