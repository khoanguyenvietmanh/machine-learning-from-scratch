import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
class Logistic_Regression:
    def __init__(self,epochs,mini_batch_size,learning_rate,LAMBDA):
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.LAMBDA = LAMBDA
    def sigmoid_function(self,z):
        return 1. / (1 + np.exp(-z))
    def compute_hypothesis(self,X_Train):
        return X_Train.dot(self.w)
    def cross_entropy_loss_function(self,X_Train,Y_Train):
        m = X_Train.shape[0]
        return (-1/m)*np.sum(Y_Train.T.dot(np.log(self.sigmoid_function(self.compute_hypothesis(X_Train)))) + (1 - Y_Train).T.dot(np.log(1-self.sigmoid_function(self.compute_hypothesis(X_Train))))) + (self.LAMBDA/(2*m))*np.linalg.norm(self.w[1:],2)**2
    def compute_gradient(self,X_Train,Y_Train):
        m = X_Train.shape[0]
        return (1/m)*np.dot((self.sigmoid_function(self.compute_hypothesis(X_Train)) - Y_Train).T,X_Train[:,0]),(1/m)*np.dot((self.sigmoid_function(self.compute_hypothesis(X_Train)) - Y_Train).T,X_Train[:,1:]).T + (self.LAMBDA/m)*self.w[1:]
    def train_using_mini_batch_gradient(self,X_Train,Y_Train):
        cross_entropy_loss = []
        for epochs in range(self.epochs):
            arr = np.array(range(X_Train.shape[0]))
            np.random.shuffle(arr)
            X_Train = X_Train[arr]
            Y_Train = Y_Train[arr]
            for i in range(0,X_Train.shape[0],self.mini_batch_size):
                X_Train_Sub = X_Train[i:i+self.mini_batch_size,:]
                Y_Train_Sub = Y_Train[i:i+self.mini_batch_size,:]

                w0_grad,w_grad = self.compute_gradient(X_Train_Sub,Y_Train_Sub)
                self.w[0]  = self.w[0] - self.learning_rate*w0_grad
                self.w[1:] = self.w[1:] - self.learning_rate*w_grad

            if np.linalg.norm(self.w,2) < 1e-6:
                    break
            cross_entropy_loss.append(self.cross_entropy_loss_function(X_Train,Y_Train))
        return cross_entropy_loss

    def init_parameter(self,X_Train):
        self.w = np.random.randn(X_Train.shape[1],1)

    def normalize_data(self,X_Train):
        X_mean = X_Train.mean(axis = 0)
        X_std = X_Train.std(axis = 0)

        X_mean = np.array([X_mean for i in range(X_Train.shape[0])])
        X_std = np.array([X_std for i in range(X_Train.shape[0])])

        return (X_Train-X_mean)/X_std

    def Load_Data_From_File(self):
        heart_disease = pd.read_csv("framingham.csv")
        heart_disease.drop(["education"],axis = 1,inplace = True)
        heart_disease.dropna(axis = 0 , inplace = True)
        X = np.array(heart_disease.iloc[:,:-1])
        Y = np.array(heart_disease.iloc[:,-1]).reshape(X.shape[0],1)
        X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size = 0.2)
        X_normalized= self.normalize_data(X_Train)
        X_Test_normalized = self.normalize_data(X_Test)
        ones = np.ones((X_normalized.shape[0],1))
        ones_2 = np.ones((X_Test_normalized.shape[0],1))
        np.concatenate([ones,X_normalized],axis = 1)
        np.concatenate([ones_2,X_Test_normalized],axis = 1)
        return X_normalized,Y_Train,X_Test_normalized,Y_Test

    def predict(self,X_Test):
        assert X_Test.shape[1] == self.w.shape[0], "Incorrect shape."
        predict = self.sigmoid_function(self.compute_hypothesis(X_Test))
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        return predict

def main():
    epochs = 100
    mini_batch_size = 20
    learning_rate = 0.01
    LAMBDA = 0.3
    LG = Logistic_Regression(epochs,mini_batch_size,learning_rate,LAMBDA)
    #LG.train_using_mini_batch_gradient()
    X_normalized,Y_Train,X_Test_normalized,Y_Test = LG.Load_Data_From_File()
    LG.init_parameter(X_normalized)
    cross_entropy_loss = LG.train_using_mini_batch_gradient(X_normalized,Y_Train)
    '''
    plt.plot(range(50),cross_entropy_loss[:50])
    plt.xlabel("iterations")
    plt.ylabel("Entropy loss")
    plt.show()
    '''
    Y_predicted = np.array(LG.predict(X_Test_normalized))
    print((len(Y_predicted[Y_predicted == Y_Test])/len(Y_Test))*100)

if __name__ == "__main__":
    main()