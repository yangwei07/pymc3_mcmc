# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:16:07 2016

@author: hrs13
"""
import numpy as np
import scipy.stats as stats
from scipy.special import psi
from scipy.special import gamma, gammaln
from example.utils import log_B

W_0 = np.eye(2) # 2D, but would work just as well for higher dimensions
v_0 = 3. # this number being higher (min is 2) makes the clusters more irregular in shape
m_0 = np.zeros(2) # clusters to be around the origin
b_0 = 0.5 # larger makes the clusters closer to the origin
a_0 = 0.4 # larger makes the mixing coefficients more similar 
    
    
def variational_E_step(X, a_k, b_k, m_k, W_k, v_k):
    N, D = X.shape
    K = len(a_k)
    r = np.empty((N, K))
    a_hat = np.sum(a_k)
    
    E_log_det = np.zeros(K)
    E_log_pis = np.zeros(K)
    for k in range(K):
        E_log_det[k] = psi(v_k[k]/2) + psi((v_k[k]+1)/2) 
        E_log_det[k] += 2 * np.log(2) + np.log(np.linalg.det(W_k[k, :, :]))
        E_log_pis[k] = psi(a_k[k]) - psi(a_hat)
        for n in range(N):    # vectorise this later
            x = np.reshape(X[n, :] - m_k[k, :], (2, 1))
            E_exponent = 2/b_k[k] + v_k[k] * x.T.dot(W_k[k, :, :]).dot(x)
            r[n, k] = np.exp(E_log_pis[k]) * np.exp(0.5*E_log_det[k]) * np.exp(-0.5*E_exponent)
    return (r.T/np.sum(r, axis=1)).T
    
def variational_LB(X, r, a_k, b_k, m_k, W_k, v_k):
    N, D = X.shape
    K = len(a_k)
    N_k = np.sum(r, axis=0)
    x_k = ((r.T.dot(X)).T/N_k).T
    X_c = np.reshape([X[n, :] - x_k[k, :] for n in range(N) for k in range(K)], (N, K, 2))
    S_k = np.einsum('nk,nkd,nke, k->kde', r, X_c, X_c, 1/N_k)
    
    a_hat = np.sum(a_k)
    
    E_log_det = np.zeros(K)
    E_log_pis = np.zeros(K)
    L = 0.
    for k in range(K):
        E_log_det[k] = psi(v_k[k]/2) + psi((v_k[k]+1)/2) 
        E_log_det[k] += 2 * np.log(2) + np.log(np.linalg.det(W_k[k, :, :]))
        E_log_pis[k] = psi(a_k[k]) - psi(a_hat)

        x = np.reshape(x_k[k, :] - m_k[k, :], (2, 1))
        L += 0.5*N_k[k]*(E_log_det[k] - 2/b_k[k] - v_k[k]*np.sum(S_k[k, :, :] * W_k[k, : :]) - v_k[k]*x.T.dot(W_k[k, :, :]).dot(x) - 2*np.log(2))

        m = np.reshape(m_k[k, :] - m_0, (2, 1))
        L += 0.5*(2*np.log(b_0/(2*np.pi)) + E_log_det[k] - 2*b_0/b_k[k] - b_0*v_k[k]*m.T.dot(W_k[k, :, :]).dot(m))
        L -= 0.5*v_k[k]*np.sum(np.linalg.inv(W_0)*W_k[k, :, :])
        
        L -= (a_k[k] - 1)*E_log_pis[k] - gammaln(a_k[k])
        
        L -= 0.5*E_log_det[k] + np.log(b_k[k]) - np.log(2*np.pi) - 1. - stats.wishart.entropy(v_k[k], W_k[k])
    L -= gammaln(a_hat)
    L += K*log_B(W_0, v_0)
    L += ((v_0-2.)/2)*np.sum(E_log_det)
    print(r)
    L -= np.sum(r*np.log(r))
    L += np.sum(r.dot(np.reshape(E_log_pis, (K, 1))))
    L += gammaln(K*a_0) - K*gammaln(a_0) + (a_0 - 1.)*np.sum(E_log_pis) 
    return float(L)

        
def variational_M_step(X, r):
    N, K = r.shape
    N_k = np.sum(r, axis=0)
    x_k = ((r.T.dot(X)).T/N_k).T
    X_c = np.reshape([X[n, :] - x_k[k, :] for n in range(N) for k in range(K)], (N, K, 2))
    S_k = np.einsum('nk,nkd,nke, k->kde', r, X_c, X_c, 1/N_k)    
    
    a_k = np.empty(K)
    b_k = np.empty(K)
    m_k = np.empty((K, 2))
    W_k = np.empty((K, 2, 2))
    v_k = np.empty(K)

    for k in range(K):    
        a_k[k] = a_0 + N_k[k]
        b_k[k] = b_0 + N_k[k]
        m_k[k, :] = (b_0 * m_0 + N_k[k] * x_k[k, :])/b_k[k]
        temp = np.linalg.inv(W_0) + N_k[k]*S_k[k, :, :]
        x = np.reshape(x_k[k, :] - m_0, (2, 1))
        temp += (b_0*N_k[k]/(b_0 + N_k[k]))*x.dot(x.T)
        W_k[k, :, :] = np.linalg.inv(temp)
        v_k[k] = v_0 + N_k[k] +1 
 
    return a_k, b_k, m_k, W_k, v_k
    
    
from example.utils import generate_parameters, generate_data
from example.plotting import double_panel_demo
from example.utils import generate_rand_samples
import os
import pandas as pd

class LoadData:
    def __init__(self):
        self.data = self.read_file()

    def read_file(self):
        header_1 = ['time', 'headway', 'ttc', 'num', 'speed']
        header_2 = ['time', 'maneuver', 'lat', 'lon', 'duration', 'th']
        time = []
        hs = []
        manu = []
        for root, dirs, file in os.walk('/home/yangwei/PycharmProjects/bayesian_analysis/UAH/D2/'):
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
        #         plt.figure()
        #         plt.plot(data_prior['x'])
        #         plt.figure()
        #         plt.plot(data_prior['dx'])
        #         plt.figure()
        #         sns.heatmap(data_prior.corr(), annot=True)
        return data_prior, m

if __name__ == '__main__':
    K = 3
    N = 1000
    num_its = 50
    # X = generate_data(N, generate_parameters(K))[0]
    d = LoadData()
    X, _ = d.prior_samples(1.)
    X = X.values
    index = np.random.choice(len(X[:, 0]), N)
    X = X[index]
    # plt = double_panel_demo(K)
    training = True
    while training:
        # X = generate_data(N, generate_parameters(K))[0]
        # plt.set_new_lims(X, num_its)
        params = generate_parameters(K)
        objective = []
        # plt.cla('ax1')
        # plt.cla('ax2')
        # plt.plot_points_black(X)
        # plt.draw()
        # plt.pause(0.001)
        rand = np.reshape(np.random.rand(N*K), (N, K))
        r = (rand.T/np.sum(rand, axis=1)).T
        for i in range(num_its):     
            
            a_k, b_k, m_k, W_k, v_k = variational_M_step(X, r)
            r = variational_E_step(X, a_k, b_k, m_k, W_k, v_k)
            
            objective.append(variational_LB(X, r, a_k, b_k, m_k, W_k, v_k))
            # plt.plot_GMM_objective(objective)
            try:
                if np.abs(objective[-1] - objective[-2]) <= 1e-1:
                    print('finish!')
                    training = False
                    break
            except:
                pass


            for i in range(10):
                params = generate_rand_samples(a_k, b_k, m_k, W_k, v_k)
                
                # plt.cla('ax1')
                # plt.plot_parameters(params)
                # plt.plot_data_coloured(X, r)
                # plt.pause(0.001)
            
   
    
    
    
#for i in range(10):
#    r = r_variational_GMM(X, a_k, b_k, m_k, W_k, v_k)
#    print LB_variational_GMM(X, r, a_k, b_k, m_k, W_k, v_k)
#
#    for j in range(10):
#        plt.cla()
#        params = generate_samples(a_k, b_k, m_k, W_k, v_k)
#        plot_parameters(params)
#        plot_data_coloured(X, r)
#        show_plot(X)
#        plt.pause(0.5)    
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    