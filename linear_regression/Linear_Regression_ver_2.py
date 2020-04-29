import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
class Linear_Regression:
    def __init__(self,epochs,learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
    def _hypothesis(self,X_Train):
        return np.dot(X_Train,self.w)
    def fit(self, X_Train, Y_Train, LAMBDA):
        assert len(X_Train.shape) == 2 and X_Train.shape[0] == Y_Train.shape[0]

        I = np.identity(X_Train.shape[1])

        self.w = np.linalg.pinv(X_Train.T.dot(X_Train) + LAMBDA*I).dot(X_Train.T).dot(Y_Train)

    def _calculate_loss_function(self,X_Train,Y_Train,LAMBDA):
        m = X_Train.shape[0]
        return (1/(2*m))*np.sum((self._hypothesis(X_Train) - Y_Train)**2) + (LAMBDA/(2*m))*np.linalg.norm(self.w[1:],2)**2

    def compute_RSS(self,Y_New,Y_Predicted):
        loss_function = (1./Y_New.shape[0])*np.sum((Y_New-Y_Predicted)**2)

        return loss_function

    def _gradient(self,X_Train,Y_Train,LAMBDA):
        m = X_Train.shape[0]
        return np.dot((self._hypothesis(X_Train) - Y_Train).T,X_Train[:,0]).T/m,np.dot((self._hypothesis(X_Train) - Y_Train).T,X_Train[:,1:]).T/m + (LAMBDA/m)*self.w[1:]

    def train(self,X_Train,Y_Train,LAMBDA):
        loss_function = []
        for epoch in range(self.epochs):
            w0_grad,w_grad = self._gradient(X_Train,Y_Train,LAMBDA)
            self.w[0] = self.w[0] - self.learning_rate * w0_grad
            self.w[1:] = self.w[1:] - self.learning_rate * w_grad
            new_loss = self.compute_RSS(Y_Train, self._hypothesis(X_Train))
            if np.linalg.norm(w_grad, 2) < 1e-6:
                break
            loss_function.append(new_loss)
        return loss_function

    def standarnize_training_set(self,X_Train):
        X_mean = X_Train.mean(axis = 0)

        X_std = X_Train.std(axis = 0)

        X_mean = np.array([X_mean for i in range(X_Train.shape[0])])
        X_std = np.array([X_std for i in range(X_Train.shape[0])])

        return (X_Train-X_mean)/X_std

    def init_parameter(self,X_Train):
        self.w = np.random.randn(X_Train.shape[1],1)

    def predict(self,X_Test):
        assert X_Test.shape[1] == self.w.shape[0], "Incorrect shape."
        return self._hypothesis(X_Test)

    def r2_score(self, y_hat, y_test):
        total_sum_squares = np.sum((y_test - np.mean(y_test))**2)
        residual_sum_squares = np.sum((y_test - y_hat)**2)
        return 1 - residual_sum_squares/total_sum_squares

    def get_the_best_LAMBDA(self,X_Train,Y_Train):
        def cross_validation(num_folds,LAMBDA):
            row_ids = np.array(range(X_Train.shape[0]))
            valid_ids = np.split(row_ids[:len(row_ids)-len(row_ids) % num_folds],num_folds)
            valid_ids.append(row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)] # D_train/D_i
            average_RSS = 0
            for i in range(num_folds):
                valid_part = {"X" : X_Train[valid_ids[i]],"Y" : Y_Train[valid_ids[i]]}
                train_part = {"X" : X_Train[train_ids[i]], "Y" : Y_Train[train_ids[i]]}
                self.fit(train_part["X"],train_part["Y"],LAMBDA)
                average_RSS+= self._calculate_loss_function(valid_part["X"],valid_part["Y"],LAMBDA)
            return average_RSS/num_folds
        def range_scan(best_LAMBDA , minimum_RSS , LAMBDA_values):
            for value in LAMBDA_values:
                average_RSS = cross_validation(5,value)
                if minimum_RSS > average_RSS:
                    minimum_RSS = average_RSS
                    best_LAMBDA = value
            return best_LAMBDA,minimum_RSS
        best_LAMBDA , minimum_RSS = range_scan(0,10000**2,range(50))

        LAMBDA_values = [k * 1./1000 for k in range(max(0,(best_LAMBDA-1)*1000),(best_LAMBDA+1)*1000,1)]

        best_LAMBDA , minimum_RSS = range_scan(best_LAMBDA,minimum_RSS,LAMBDA_values)

        return best_LAMBDA
    def plot_loss_function(self,loss_function):
        i = range(len(loss_function))

        plt.ylabel("Loss function")

        plt.xlabel("Number of epochs")

        plt.plot(i, loss_function, color="r")

        plt.show()


def main():
    X = np.loadtxt("prostate.data.txt",skiprows=1)
    learning_rate = 0.01
    epochs = 1000
    linear_regression = Linear_Regression(epochs, learning_rate)
    #X_normalized = normalize_and_add_one(X)
    X_normalized = linear_regression.standarnize_training_set(X)
    ones = np.ones((X_normalized.shape[0],1))
    X_normalized = np.concatenate([ones,X_normalized],axis = 1)
    X_normalized = np.array(X_normalized[:,:-1])
    Y_Train_1 = np.array(X[:,-1]).reshape(X.shape[0],1)

    X_Train, Y_Train = X_normalized[:50], Y_Train_1[:50]
    X_Test, Y_Test = X_normalized[50:], Y_Train_1[50:]

    linear_regression.init_parameter(X_normalized)

    best_LAMBDA = linear_regression.get_the_best_LAMBDA(X_Train,Y_Train)

    print(f"best LAMBDA:{best_LAMBDA}")

    loss_function = linear_regression.train(X_Train,Y_Train,best_LAMBDA)

    print(f"Error of Predicted set and Test set:{linear_regression._calculate_loss_function(X_Test,Y_Test,best_LAMBDA)}")

    linear_regression.plot_loss_function(loss_function)

main()




