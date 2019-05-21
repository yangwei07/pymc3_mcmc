import os
import pickle
import torch
import torch.nn as nn
import scipy.stats
import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import theano.tensor as tt
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from itertools import permutations
from sklearn.mixture import GaussianMixture

K = 3
ON_TRAIN = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Functions:
    def __init__(self):
        pass

    def make_cov_matrix(self, sigma=None, corr=None, chol=None, module=np):
        if chol is None:
            C = module.ones((2, 2))
            var = module.diag(sigma)
            idxu = np.triu_indices(2, 1)
            idxl = np.tril_indices(2, -1)
            if module == np:
                C[idxu] = corr
                C[idxl] = corr
            else:
                C = tt.set_subtensor(C[idxu], corr)
                C = tt.set_subtensor(C[idxl], corr)
            cov = var.dot(C).dot(var)
        else:
            C = module.zeros((2, 2))
            idxl = np.mask_indices(2, np.tril)
            if module == np:
                C[idxl] = chol
            else:
                C = tt.set_subtensor(C[idxl], chol)
            cov = C.dot(C.T)
        return cov

    def sampling(self, w, mu, cov, N=1000):
        num = int(w * N)
        if len(mu) == 1:
            samples = np.random.normal(mu, cov, size=num)
        else:
            samples = np.random.multivariate_normal(mu, cov, size=num)
        return samples

    def flatten(self, input_list):
        output_list = []
        while True:
            if input_list == []:
                break
            for index, i in enumerate(input_list):
                if type(i) == list:
                    input_list = i + input_list[index + 1:]
                    break
                else:
                    output_list.append(i)
                    input_list.pop(index)
                    break
        return output_list

    def norm_1d(self, x, mu, sigma):
        prob = np.zeros(x.shape[0])
        for i, v in enumerate(x):
            prob[i] = 1 / (np.sqrt(2 * 3.14 * sigma ** 2)) * np.exp(-0.5 * (v - mu) ** 2 / sigma ** 2)
        return prob

    def norm_2d(self, x, w, mu, cov):
        prob = np.zeros(x.shape[0])
        for i, v in enumerate(x):
            c = [np.sqrt(1 / ((2 * 3.14) ** 2 * np.linalg.det(cov[k]))) for k in range(K)]
            f = [c[k] * np.exp(-0.5 * (v - mu[k]).dot(np.linalg.inv(cov[k])).dot((v - mu[k]).T)) for k in range(K)]
            prob[i] = np.sum([w[k] * f[k] for k in range(K)])
        return prob

    def cluster(self, x, w, mu, cov):
        prob = np.zeros([x.shape[0], K])
        cluster = np.zeros(x.shape[0])
        for i, v in enumerate(x):
            c = [np.sqrt(1 / ((2 * 3.14) ** 2 * np.linalg.det(cov[k]))) for k in range(K)]
            f = [c[k] * np.exp(-0.5 * (v - mu[k]).dot(np.linalg.inv(cov[k])).dot((v - mu[k]).T)) for k in range(K)]
            for k in range(K):
                prob[i, k] = w[k] * f[k]
            cluster[i] = prob[i].argmax()
        return cluster

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def stocGraAscent(self, data, label):
        m, n = np.shape(data)
        w = np.ones([n, 1])
        alpha = 0.25
        gamma = 2
        episode = 1000
        mini_batch = 1000
        w_trace = [w]
        loss = []
        for i in range(episode):
            dataIndex = np.random.choice(m, mini_batch, False)
            lr = 0.001
            pred = self.sigmoid(data[dataIndex].dot(w)).T
            real = label[dataIndex]
            # error = self.sigmoid(data[dataIndex].dot(w)).T - label[dataIndex]
            error = -real * alpha * (1 - pred) ** gamma * np.log(pred+1e-5) - \
                    (1 - real) * (1 - alpha) * pred ** gamma * np.log(1 - pred+1e-5)
            crossentropyloss = np.sum([-y * np.log(x) - (1 - y) * np.log(1 - x) for x, y in zip(pred, real)])
            # if mse <= 1e-3:
            #     break
            # else:
            w = w - lr * data[dataIndex].T.dot(error.T)
            w_trace.append(w)
            loss.append(crossentropyloss)
        w_trace = np.hstack(w_trace)
        return w_trace


class LoadData:
    def __init__(self):
        self.data = self.read_file()

    def read_file(self):
        header_1 = ['time', 'headway', 'ttc', 'num', 'speed']
        header_2 = ['time', 'maneuver', 'lat', 'lon', 'duration', 'th']
        time = []
        hs = []
        manu = []
        for root, dirs, file in os.walk('./UAH/D1/'):
            for path in dirs:
                detection = pd.read_csv(os.path.join(root, path, 'PROC_VEHICLE_DETECTION.txt'), sep='\s+', names=header_1)
                event = pd.read_csv(os.path.join(root, path, 'EVENTS_LIST_LANE_CHANGES.txt'), sep='\s+', names=header_2)
                time_1 = np.around(detection['time'].values, 1)
                time_2 = np.around(event['time'].values, 1)
                headway = detection['headway'].values
                t = np.arange(time_1[0], time_1[-1], 0.1)
                s = np.interp(t, time_1, headway)
                index = []
                for d in time_2:
                    index.append(np.where(np.abs(t - d) <= 1e-3)[0][0])
                m = np.zeros_like(t)
                m[index] = np.sign(event['maneuver'].values)
                time.append(t)
                hs.append(s)
                manu.append(m)
        time = np.concatenate(time)
        hs = np.concatenate(hs)
        manu = np.concatenate(manu)
        x0, dx0, m0 = self.grad(time, hs, manu)
        return x0, dx0, m0

    def grad(self, t, x, m):
        def outlier(s, x, level=1):
            mean_x = np.mean(s)
            std_x = np.std(s)
            index = np.where(np.abs(s - mean_x) >= level * std_x)
            x = [np.delete(x[k], index) for k in range(len(x))]
            return x

        def kalman(obs, x, p=np.eye(2), dt=0.1):
            eta = []
            for i in range(len(obs)):
                A = np.array([[1, dt], [0, 1]])
                C = np.array([[1, 0]])
                Q = 100 * np.eye(2)
                R = 1
                x = A.dot(x)
                p = A.dot(p).dot(A.T) + Q
                k = p.dot(C.T).dot(np.linalg.inv(C.dot(p).dot(C.T) + R))
                x = x + k.dot(obs[i] - C.dot(x))
                p = (np.eye(2) - k.dot(C)).dot(p)
                eta.append(x[1])
            return np.array(eta)

        headway = []
        speed = []
        maneuver = []
        i_0 = []
        i_f = []
        for i in range(len(x) - 1):
            if x[i] == -1 and x[i + 1] != -1:
                i_0.append(i)
            if x[i] != -1 and x[i + 1] == -1:
                i_f.append(i)
        num = np.min([len(i_0), len(i_f)])
        for i in range(num):
            if len(x[i_0[i] + 1: i_f[i] + 1]) < 2:
                continue
            s = x[i_0[i] + 1: i_f[i] + 1]
            ds1 = np.gradient(x[i_0[i] + 1: i_f[i] + 1], t[i_0[i] + 1: i_f[i] + 1])
            ds2 = kalman(s, np.array([s[0], ds1[0]]))
            headway.append(s)
            speed.append(ds2)
            maneuver.append(m[i_0[i] + 1: i_f[i] + 1])
        headway = np.concatenate(headway)
        speed = np.concatenate(speed)
        maneuver = np.concatenate(maneuver)
        headway, speed, maneuver = outlier(headway, (headway, speed, maneuver), 1)
        headway, speed, maneuver = outlier(speed, (headway, speed, maneuver), 1)
        return headway, speed, maneuver

    def prior_samples(self, percent=0.1):
        data_size = len(self.data[0])
        num = int(data_size * percent)
        index = np.random.choice(data_size, num, False)
        # index = range(num)
        x = self.data[0][index]
        dx = self.data[1][index]
        m = self.data[2][index]
        data_prior = pd.DataFrame([x, dx]).transpose()
        data_prior = data_prior.dropna(axis=0)
        data_prior.columns = ['x', 'dx']
        return data_prior, m


class BayesianGMM(Functions):
    def __init__(self, data):
        super(BayesianGMM, self).__init__()
        self.model = pm.Model()
        self.modeling(data)

    def modeling(self, data):
        with self.model:
            w = pm.Dirichlet('w', a=np.ones(K))
            mu = [pm.Normal('mu'+str(k), mu=[20, 0], sd=10, shape=2) for k in range(K)]
            sigma = [pm.HalfNormal('sigma'+str(k), sd=5, shape=2) for k in range(K)]
            corr = [pm.Uniform('corr'+str(k), lower=-1, upper=1, shape=1) for k in range(K)]
            cov = [self.make_cov_matrix(sigma=sigma[k], corr=corr[k], module=tt) for k in range(K)]
            dist = [pm.MvNormal.dist(mu=mu[k], cov=cov[k]) for k in range(K)]
            obs = pm.Mixture('obs', w=w, comp_dists=dist, observed=data)

    def training(self, advi=False):
        with self.model:
            if advi:
                inference = pm.fit(method='advi')
                trace = inference.sample()
            else:
                trace = pm.sample(chains=1)
        pm.summary(trace)
        return trace

    def plot(self, trace, type=0):
        with self.model:
            if type == 0:
                pm.traceplot(trace)
            elif type == 1:
                pm.plot_posterior(trace)
            elif type == 2:
                pm.forestplot(trace)
            elif type == 3:
                pm.autocorrplot(trace)
            elif type == 4:
                pm.energyplot(trace)
            elif type == 5:
                pm.densityplot(trace)
            else:
                pass


class PostProcess(Functions):
    def __init__(self, data, trace):
        super(PostProcess, self).__init__()
        self.data = data
        self.trace = trace
        self.var, self.var_mu, self.var_std, self.var_name = self.stats()
        self.gmm = GaussianMixture(K).fit(data)
        if K==3:
            colours = np.eye(3)
        else:
            colours = dirichlet.rvs(0.1*np.ones(3), K) + dirichlet.rvs(0.1*np.ones(3), K)
            colours = colours / sum(colours)
        self.colours = colours
        self.z_to_colour = lambda z: colours.T.dot(np.reshape(z, (K, 1)))

    def stats(self):
        order = np.argsort(self.trace['w'].mean(0))
        var_name = [['mu'+str(k) for k in order], ['sigma'+str(k) for k in order], ['corr'+str(k) for k in order]]
        var_name = self.flatten(var_name)
        var = [np.vstack([self.trace['w'][:, k] for k in order]).T]
        var_mu = [var[0].mean(0)]
        var_std = [var[0].std(0)]

        for name in var_name:
            var.append(self.trace[name])
            var_mu.append(self.trace[name].mean(0))
            var_std.append(self.trace[name].std(0))
        var_name.insert(0, 'w')
        return var, var_mu, var_std, var_name

    def post_samples(self, N=1000):
        w = [self.var_mu[0][k] for k in range(K)]
        mu = [self.var_mu[k + 1] for k in range(K)]
        cov = [self.make_cov_matrix(sigma=self.var_mu[k + 1 + K], corr=self.var_mu[k + 1 + 2 * K]) for k in range(K)]
        samples = [self.sampling(w[k], mu[k], cov[k], N) for k in [2, 0, 1]]
        fig = plt.figure()
        plt.subplot(121)
        [plt.scatter(samples[k][:, 0], samples[k][:, 1], label='component'+str(k)) for k in range(K)]
        plt.legend()
        plt.xlim([5, 45])
        plt.ylim([-30, 30])
        plt.subplot(122)
        data_post = np.concatenate(samples)
        p = self.cluster(data_post, w, mu, cov)
        plt.scatter(data_post[:, 0], data_post[:, 1], c=p)
        # plt.legend()
        plt.xlim([5, 45])
        plt.ylim([-30, 30])
        data_post = pd.DataFrame(data_post)
        data_post.columns = ['x', 'dx']
        fig.savefig('./img/post_samples.svg', format='svg')
        return data_post

    def joint_dist(self, data_post):
        # fig = plt.figure()
        sns.jointplot('x', 'dx', data=self.data, xlim=[5, 45], ylim=[-30, 30], kind='kde', space=0, color='r')
        sns.jointplot('x', 'dx', data=data_post, xlim=[5, 45], ylim=[-30, 30], kind='kde', space=0, color='g')
        # fig.savefig('./img/joint_dist.svg', format='svg')

    def marginalized_dist(self, data_post):
        fig = plt.figure()
        plt.subplot(121)
        sns.distplot(self.data['x'], bins=30, kde=False, norm_hist=True,
                     hist_kws={'histtype': 'step', 'linewidth': 3}, label='Prior distribution')
        sns.distplot(data_post['x'], bins=30, kde=False, norm_hist=True,
                     hist_kws={'histtype': 'step', 'linewidth': 3}, label='Posterior distribution')
        plt.legend()
        plt.xlim([5, 45])
        plt.subplot(122)
        sns.distplot(self.data['dx'], bins=30, kde=False, norm_hist=True,
                     hist_kws={'histtype': 'step', 'linewidth': 3}, label='Prior distribution')
        sns.distplot(data_post['dx'], bins=30, kde=False, norm_hist=True,
                     hist_kws={'histtype': 'step', 'linewidth': 3}, label='Posterior distribution')
        plt.legend()
        plt.xlim([-30, 30])
        fig.savefig('./img/marginalized_dist.svg', format='svg')

    def ordered(self):
        order = list(permutations(range(K)))
        mse = np.zeros(len(order))
        list_order = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9]
        for n, l in enumerate(order):
            vars = [self.gmm.weights_]
            for k in l:
                vars.append(self.gmm.means_[k])
                sdx = np.sqrt(self.gmm.covariances_[k][0][0])
                sdy = np.sqrt(self.gmm.covariances_[k][1][1])
                vars.append(np.array([sdx, sdy]))
                corr = np.array([self.gmm.covariances_[k][1][0] / (sdx * sdy)])
                vars.append(corr)
            gmm_map = [vars[k] for k in list_order]
            value = 0.
            for i in range(len(gmm_map)):
                for (x, y) in zip(self.var_mu, gmm_map):
                    value += np.sum((x - y) ** 2)
            mse[n] = value
        index = mse.argmin()
        vars = [self.gmm.weights_]
        for k in order[index]:
            vars.append(self.gmm.means_[k])
            sdx = np.sqrt(self.gmm.covariances_[k][0][0])
            sdy = np.sqrt(self.gmm.covariances_[k][1][1])
            vars.append(np.array([sdx, sdy]))
            corr = np.array([self.gmm.covariances_[k][1][0] / (sdx * sdy)])
            vars.append(corr)
        gmm_map = [vars[k] for k in list_order]
        return gmm_map

    def compare_1d(self):
        gmm_map = self.ordered()
        fig = plt.figure()
        for i in range(len(gmm_map)):
            plt.subplot2grid([len(gmm_map), 2], [i, 0])
            plt.ylabel('Frequency')
            plt.title(self.var_name[i])
            col = self.var[i].shape[1]
            x = [np.linspace(self.var[i][:, k].min(), self.var[i][:, k].max(), 50) for k in range(col)]
            p = [self.norm_1d(x[k], self.var_mu[i][k], self.var_std[i][k]) for k in range(col)]
            p_max = [p[k].max() for k in range(col)]
            [plt.plot(x[k], p[k]) for k in range(col)]
            [plt.plot([gmm_map[i][k], gmm_map[i][k]], [0, p_max[k]], color='r') for k in range(col)]
            plt.subplot2grid([len(gmm_map), 2], [i, 1])
            plt.ylabel('Sample value')
            plt.title(self.var_name[i])
            plt.plot(self.var[i])
        fig.savefig('./img/compare_1d.svg', format='svg')

    def compare_2d(self):
        w = [self.var_mu[0][k] for k in range(K)]
        mu = [self.var_mu[k + 1] for k in range(K)]
        cov = [self.make_cov_matrix(sigma=self.var_mu[k + 1 + K], corr=self.var_mu[k + 1 + 2 * K]) for k in range(K)]
        p1 = self.norm_2d(self.data.values, self.gmm.weights_, self.gmm.means_, self.gmm.covariances_)
        p2 = self.norm_2d(self.data.values, w, mu, cov)
        p1 = p1 / np.sum(p1)
        p2 = p2 / np.sum(p2)
        # fig = plt.figure()
        # plt.subplot(121)
        # plt.scatter(self.data['x'], self.data['dx'], c=p1)
        # plt.colorbar()
        # plt.xlim([5, 45])
        # plt.ylim([-30, 30])
        # plt.subplot(122)
        # plt.scatter(self.data['x'], self.data['dx'], c=p2)
        # plt.colorbar()
        # plt.xlim([5, 45])
        # plt.ylim([-30, 30])
        # fig.savefig('./img/compare_2d.svg', format='svg')
        return p1

    def step(self):
        plt.figure(1)
        for i in range(len(self.var)):
            plt.subplot2grid([len(self.var), 1], [i, 0])
            plt.ylabel('Frequency')
            plt.title(self.var_name[i])
            col = self.var[i].shape[1]
            x = [np.linspace(self.var[i][:, k].min(), self.var[i][:, k].max(), 50) for k in range(col)]
            p = [self.norm_1d(x[k], self.var_mu[i][k], self.var_std[i][k]) for k in range(col)]
            [plt.plot(x[k], p[k]) for k in range(col)]


class Estimation(PostProcess):
    def __init__(self, data, m, trace):
        super(Estimation, self).__init__(data, trace)
        self.data = data
        self.m = m

    def importance_sampling(self, prob):
        N = 5000
        index = np.random.choice(self.data.shape[0], N, True, prob)
        samples = self.data.iloc[index]
        s = self.m[index]
        fig = plt.figure()
        sns.scatterplot(x='x', y='dx', data=samples, alpha=0.1)
        plt.scatter(samples['x'].iloc[s != 0], samples['dx'].iloc[s != 0], label='lane change')
        plt.xlim([5, 45])
        plt.ylim([-10, 10])
        plt.legend()
        fig.savefig('./img/importance_sampling.svg', format='svg')
        print(samples['x'].iloc[s != 0].shape[0] / N)

    def maneuver(self):
        N = self.data.shape[0]
        # index = np.random.choice(self.data.shape[0], N, True, prob)
        index = range(self.data.shape[0])
        samples = self.data.iloc[index]
        s = self.m[index]
        w = [self.var_mu[0][k] for k in range(K)]
        mu = [self.var_mu[k + 1] for k in range(K)]
        cov = [self.make_cov_matrix(sigma=self.var_mu[k + 1 + K], corr=self.var_mu[k + 1 + 2 * K]) for k in range(K)]
        c = self.cluster(samples.values, w, mu, cov)
        fig = plt.figure()
        for i in range(3):
            plt.subplot(131+i)
            plt.scatter(samples['x'].iloc[c==i], samples['dx'].iloc[c==i], alpha=0.1)
            index1 = np.where(c == i)[0]
            index2 = np.where(s != 0)[0]
            index = list(set(index1) & set(index2))
            plt.scatter(samples['x'].iloc[index], samples['dx'].iloc[index], c='r')
        # plt.scatter(samples['x'], samples['dx'], alpha=0.1)
        # plt.scatter(samples['x'].iloc[s != 0], samples['dx'].iloc[s != 0], c=c[s != 0])
            plt.xlim([5, 45])
            plt.ylim([-30, 30])
        fig.savefig('./img/importance_sampling.svg', format='svg')
        print(samples['x'].iloc[s != 0].shape[0] / N)

    def logit_classify(self):
        w_trace = self.stocGraAscent(self.data.values, self.m)
        return w_trace


if __name__ == '__main__':
    if ON_TRAIN:
        # PERCENT = np.arange(0.1, 1., 0.1)
        # PERCENT = np.linspace(0.1, 0.1, 10)
        PERCENT = [0.8]
        data = []
        d = LoadData()
        data_prior, m = d.prior_samples(PERCENT[0])
        for p in PERCENT:
            f = BayesianGMM(data_prior)
            trace = f.training()
            data.append([data_prior, m, trace])
        with open('./results/data6.pkl', 'wb') as file:
            pickle.dump(data, file)
    else:
        with open('./results/data6.pkl', 'rb') as file:
            data = pickle.load(file)
        for i in range(1): #range(len(data)):
            data_prior, m, trace = data[i]
            p = PostProcess(data_prior, trace)
            # data_post = p.post_samples(data_prior.shape[0])
            # p.joint_dist(data_post)
            # p.marginalized_dist(data_post)
            # p.compare_1d()
            prob = p.compare_2d()
            e = Estimation(data_prior, m, trace)
            # w_trace = e.logit_classify()
            # w = w_trace[:, -1]
            # pred = e.sigmoid(data_prior.dot(w)).T
            # plt.figure()
            # plt.subplot(211)
            # plt.plot(pred, c='r', label='predict')
            # plt.legend()
            # plt.subplot(212)
            # plt.plot(m, c='b', label='truth')
            # plt.legend()
            # plt.figure()
            # plt.scatter(data_prior['x'], data_prior['dx'])
            # plt.scatter(data_prior['x'].iloc[m!=0], data_prior['dx'].iloc[m!=0])
            e.maneuver()
            # e.importance_sampling(prob)
            # p.step()
            # if data_prior.shape[0] > data_post.shape[0]:
            #     num = data_post.shape[0]
            #     data_prior = data_prior.iloc[:num]
            # if data_prior.shape[0] < data_post.shape[0]:
            #     num = data_prior.shape[0]
            #     data_post = data_post.iloc[:num]
            # KL = scipy.stats.entropy(data_prior.values[:, 0], data_post.values[:, 0])
            # print(KL)