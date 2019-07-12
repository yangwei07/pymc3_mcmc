import pymc3 as pm
import numpy as np
import scipy as sp
import pandas as pd
import theano

import matplotlib.pyplot as plt
data = pd.read_csv('speed.txt', sep='\s+', names=['time', 'speed'])

num = np.arange(0, 1000, 1)
time = data['time'].values.reshape(-1, 1)[num]
speed = data['speed'].values[num]

plt.figure()
plt.scatter(time, speed)
plt.show()
with pm.Model() as model:
    l = pm.Gamma('l', alpha=2, beta=1)
    n = pm.HalfCauchy('n', beta=1)

    inverse_lengthscale = 5
    # cov = pm.gp.cov.Exponential(1, ls_inv=inverse_lengthscale)
    cov = n ** 2 * pm.gp.cov.Matern32(1, l)
    gp = pm.gp.Marginal(cov_func=cov)

    s = pm.HalfCauchy('s', beta=5)
    y_ = gp.marginal_likelihood('y', X=time, y=speed, noise=s)

    mp = pm.find_MAP()

# with pm.Model() as model:
#     l = pm.Gamma('l', alpha=2, beta=1)
#     n = pm.HalfCauchy('n', beta=5)
#
#     cov = n ** 2 * pm.gp.cov.Matern52(1, l)
#     gp = pm.gp.Latent(cov_func=cov)
#
#     f = gp.prior("f", X=time)
#
#     s = pm.HalfCauchy('s', beta=5)
#     v = pm.Gamma('v', alpha=2, beta=0.1)
#     y_ = pm.StudentT('y', mu=f, lam=1.0/s, nu=v, observed=speed)
#
#     mp = pm.sample(1000, chains=1)


num = np.arange(0, 5000, 50)
X_new = data['time'].values.reshape(-1, 1)[num]
with model:
    f_pred = gp.conditional("f_pred", X_new)
    pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=10)

from pymc3.gp.util import plot_gp_dist
fig = plt.figure()
ax = fig.gca()
plot_gp_dist(ax, pred_samples["f_pred"], X_new)
ax.scatter(data['time'].values.reshape(-1, 1)[:5000], data['speed'].values[:5000])
plt.show()