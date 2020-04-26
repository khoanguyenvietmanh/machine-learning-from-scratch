import numpy as np
def Read_DataSet_File():
    content = np.loadtxt("Data_Set_Regression.txt")
    return content
def normalize_and_add_one(X):
    X  = np.array(X)

    X_Max  = np.array([[np.amax(X[:,column_id]) for column_id in range(X.shape[1])] for i in range(X.shape[0])])

    X_Min = np.array([[np.amin(X[:, column_id]) for column_id in range(X.shape[1])] for i in range(X.shape[0])])

    X_normalized = (X - X_Min)/(X_Max-X_Min)

    ones = np.ones(X.shape[0])

    X_normalized[:,0] = ones

    return X_normalized
class Ridge_Regression:
    def __init__(self):
        return
    def fit(self, X_Train, Y_Train, LAMBDA):
        assert len(X_Train) == 2 or X_Train.shape[0] == Y_Train.shape[0]

        I = np.identity(X_Train.shape[1])

        W = np.linalg.pinv(X_Train.T.dot(X_Train) + LAMBDA*I).dot(X_Train.T).dot(Y_Train)

        return W

    def compute_RSS(self,Y_New,Y_Predicted):
        loss_function = (1./Y_New.shape[0])*np.sum(np.square(Y_New-Y_Predicted))

        return loss_function
    def predict(self,W,X_New):
        Y_New = X_New.dot(W)

        return Y_New
    def fit_gradient(self,X_Train,Y_Train,LAMBDA,learning_rate,max_num_epochs=100,mini_batch_size = 30):
        W = np.random.randn(X_Train.shape[1])
        last_loss = 10e+8
        for ep in range(max_num_epochs):
            arr = np.random.shuffle(range(X_Train.shape[0]))
            X_Train = X_Train[arr]
            Y_Train = Y_Train[arr]
            for i in range(0,X_Train.shape[0],mini_batch_size):
                X_Train_Sub = X_Train[i:i+mini_batch_size]

                Y_Train_Sub = Y_Train[i:i+mini_batch_size]

                grad = np.dot((W.dot(X_Train_Sub) - Y_Train_Sub),X_Train_Sub) + LAMBDA * W

                W = W - learning_rate*grad

            new_loss = self.compute_RSS((self.predict(W,X_Train),Y_Train))
            if np.abs(new_loss - last_loss) <= 1e-3:
                break
            last_loss = new_loss
        return W
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
                W = self.fit(train_part["X"],train_part["Y"],LAMBDA)
                Y_Predicted = self.predict(W,valid_part["X"])
                average_RSS+= self.compute_RSS(valid_part["Y"],Y_Predicted)
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

if __name__ == "__main__":
    content = Read_DataSet_File()
    X_normalized = normalize_and_add_one(content)
    Y_normalized = np.array(X_normalized[:, -1]).reshape(X_normalized.shape[0], 1)
    X_normalized = np.array(X_normalized[:,:-1])
    X_Train, Y_Train = X_normalized[:50], Y_normalized[:50]
    X_Test, Y_Test = X_normalized[50:], Y_normalized[50:]
    rrg = Ridge_Regression()
    best_LAMBDA = rrg.get_the_best_LAMBDA(X_Train, Y_Train)
    print(f"Best LAMBDA : {best_LAMBDA}")
    W_learned = rrg.fit(X_Train, Y_Train, best_LAMBDA)
    Y_predicted = rrg.predict(W_learned, X_Test)
    print(rrg.compute_RSS(Y_Test, Y_predicted))