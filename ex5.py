import numpy as np
import matplotlib.pyplot as plt


def generateData():
    mean = np.array([1., 2.])
    cov = np.array([[1., 0.9], [0.9, 1.]])
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    X = np.stack([np.ones_like(x), x], axis=1)
    return X, y

def phi(x, d):
    # a
    return x.reshape(x.shape[0], 1) ** np.arange(d).reshape(1, d)

def posterior_w(X, y, w0, V0, Sigma):
    invV0    = np.linalg.inv(V0)
    invSigma = np.linalg.inv(Sigma)
    Vn = np.linalg.inv(X.T @ invSigma @ X + invV0)
    wn = Vn @ (invV0 @ w0 + X.T @ invSigma @ y)
    return wn, Vn


def doTask2():
    x, y = generateData()
    print(x.size)
    d = 2
    w0 = np.zeros(d)
    V0 = np.eye(d)
    Sigma = np.eye(d)
    wn, Vn = posterior_w(x, y, w0, V0, Sigma)

    print(wn)



if __name__ == '__main__':
    doTask2()
