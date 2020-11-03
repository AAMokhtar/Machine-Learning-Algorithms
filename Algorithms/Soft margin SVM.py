import cvxopt
import numpy as np


def linear_kernel(X1, X2):
    return np.dot(X1, X2)

def polynomial_kernel(X, Y, p=3):
    return (1 + np.dot(X, Y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

class SVM:
    def __init__(self, kernel=linear_kernel, C=None,p=3,sigma=5.0):
        self.kernel = kernel
        self.p = p
        self.sigma = float(sigma)
        self.C = C
        self.A = None
        self.SV = None
        self.SV_Y = None
        self.W = None
        self.b = None

        # convert int values
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, Y):
        samples_n, features_n = X.shape

        # kernel trick for all training pairs K(xi*xj)
        K = np.zeros((samples_n, samples_n))
        for i in range(samples_n):
            for j in range(samples_n):
                if self.kernel == polynomial_kernel:
                    K[i, j] = self.kernel(X[i], X[j], self.p)
                elif self.kernel == gaussian_kernel:
                    K[i, j] = self.kernel(X[i], X[j], self.sigma)
                else:
                    K[i, j] = self.kernel(X[i], X[j])

        # manipulate the dual form to conform to the standard CVXOPT QP form
        # min x: (1/2)x.T(p)x + (q).Tx
        # provide: p, q
        p = cvxopt.matrix(np.outer(Y, Y) * K)
        q = cvxopt.matrix(np.ones(samples_n) * -1)
        # s.t: (G)x <= (h) , (A)x = (b)
        # provide: G, h, A, b

        # CONDITION: C = None -> C = inf -> Hard margin SVM
        # ai >= 0, i = 0,1,2...
        if self.C is None:
            G = cvxopt.matrix(np.identity(samples_n) * -1 + 0)
            h = cvxopt.matrix(np.zeros(samples_n))
        # ALTERNATE CONDITION: C != None -> Soft margin SVM
        # 0 =< ai <= C, i = 0,1,2...
        # Broken down into: -ai <= 0, ai <= C
        else:
            G = cvxopt.matrix(np.vstack((np.identity(samples_n) * -1 + 0, np.identity(samples_n))))
            h = cvxopt.matrix(np.hstack((np.zeros(samples_n), np.ones(samples_n) * self.C)))

        A = cvxopt.matrix(Y, (1, samples_n))
        b = cvxopt.matrix(0.0)

        # Solve problem
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(p, q, G, h, A, b)

        # extract lagrange multipliers
        all_A = np.ravel(sol['x'])

        # filter support vectors (non zero lagrange multipliers -> lie on margins)
        # and save them
        indices_bool = all_A > 1e-5
        all_A = all_A.reshape(samples_n,1)
        self.A = all_A[indices_bool]
        self.SV = X[indices_bool]
        self.SV_Y = Y[indices_bool]

        # weights
        # if kernel != linear we calculate the weights when we predict in order to
        # apply the kernel
        AxY = self.A * self.SV_Y
        if self.kernel == linear_kernel:
            self.W = np.dot(self.SV.T, AxY)

        # Intercept
        k_rows = K[indices_bool]
        self.b = np.sum(self.SV_Y) - np.sum(AxY * k_rows[:, indices_bool])
        self.b /= len(self.A)

    # vector from the boundary to each sample in X
    def project(self, X):

        # if linear kernel
        if self.W is not None:
            return np.dot(X, self.W) + self.b

        # no vectorization for gaussian kernel
        if self.kernel == gaussian_kernel:
            k_arr = np.zeros((X.shape[0], self.SV.shape[0]))
            for i in range(X.shape[0]):
                for j in range(self.SV.shape[0]):
                    k_arr[i][j] = self.kernel(X[i], self.SV[j], self.sigma)
        # polynomial kernel
        else:
            k_arr = self.kernel(X, self.SV.T, self.p)

        return np.sum((self.A * self.SV_Y).T * k_arr, axis=1).reshape(X.shape[0],1)

    def predict(self, X):
        # -1 or 1
        return np.sign(self.project(X))

# driver code
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    data = pd.read_csv("Data/Social_Network_Ads.csv")

    # separate response variable
    Y = data["Purchased"]

    # drop the response variable and id columns
    data.drop(columns=["User ID", "Purchased"], inplace=True)

    # replace gender with a dummy variable
    data["Gender"] = pd.Categorical(data["Gender"])
    data["Gender"] = data["Gender"].cat.codes

    # 80% train, 20% test
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size=0.2, random_state=53)

    X_train = X_train.values.reshape(X_train.shape[0], -1)
    X_train = preprocessing.scale(X_train)

    X_test = X_test.values.reshape(X_test.shape[0], -1)
    X_test = preprocessing.scale(X_test)

    Y_train = np.array(list(map(lambda e: 1.0 if e == 1 else -1.0, Y_train))).reshape(Y_train.shape[0], 1)
    Y_test = np.array(list(map(lambda e: 1.0 if e == 1 else -1.0, Y_test))).reshape(Y_test.shape[0], 1)

    cls = SVM(kernel=polynomial_kernel, p=5, C=10)
    cls.fit(X_train.astype(float),Y_train)

    predictions_test = cls.predict(X_test)
    accuracy = np.sum(np.abs(predictions_test + Y_test)) / (2 * Y_test.shape[0])
    print("test accuracy = %2.2f%%" % (accuracy * 100))