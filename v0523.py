import numpy as np
import matplotlib.pyplot as plt

w = [0.3, 0.7]
mu = [[20, 15], [1, 2]]
sdx = [2, 1]
sdy = [1, 3]
corr = [0.1, -0.05]

class Utils:
    def __init__(self):
        pass

    def make_cov_matrix(self, sdx, sdy, corr):
        cov = []
        for i in range(len(sdx)):
            C = np.ones((2, 2))
            var = np.diag([sdx[i], sdy[i]])
            idxu = np.triu_indices(2, 1)
            idxl = np.tril_indices(2, -1)
            C[idxu] = corr[i]
            C[idxl] = corr[i]
            cov.append(var.dot(C).dot(var))
        return cov


cov = Utils().make_cov_matrix(sdx, sdy, corr)
plt.figure()
for i in range(2):
    num = int(w[i] * 1000)
    data = np.random.multivariate_normal(mu[i], cov[i], num)
    plt.scatter(data[:, 0], data[:, 1], label='component'+str(i))
plt.legend()
