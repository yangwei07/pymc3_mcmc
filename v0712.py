import numpy as np
# from scipy.interpolate import spline
from scipy.linalg import cho_solve
from numpy.linalg import cholesky
from itertools import cycle
import pandas as pd
import matplotlib.pyplot as plt

class SimpleGP():
    """ One dimensional Gaussian Process class.  Uses
    squared exponential covariance form.

    parameters
    ----------
    width_scale : float, positive
        Same as sigma in (4) of post

    length_scale : float, positive
        Same as l in (4) of post

    noise : float
        Added to diagonal of covariance, useful for
        improving convergence
    """

    def __init__(self, width_scale, length_scale, noise=10 ** (-6)):
        self.width_scale = width_scale
        self.length_scale = length_scale
        self.noise = noise

    def _exponential_cov(self, x1, x2):
        """
        Return covariance matrix for two arrays,
        with i-j element = cov(x_1i, x_2j).

        parameters
        ----------
        x1, x2: np.array
            arrays containing x locations
        """
        return (self.width_scale ** 2) * np.exp(
            - np.subtract.outer(x1, x2) ** 2 / (2 * self.length_scale ** 2))

    def fit(self, sample_x, sample_y, sample_s):
        """
        Save for later use the Cholesky matrix
        associated with the inverse that appears
        in (5) of post. Also evaluate the weighted
        y vector that appears in that equation.

        parameters
        ----------
        sample_x : np.array
            locations where we have sampled

        sample_y : np.array
            y values observed at each sample location

        sample_s : np.array
            array of stds for each sample
        """

        self.sample_x = np.array(sample_x)

        S = self._exponential_cov(sample_x, sample_x)
        d = np.diag(np.array(sample_s) ** 2 + self.noise)

        self.lower_cholesky = cholesky(S + d)
        self.weighted_sample_y = cho_solve(
            (self.lower_cholesky, True), sample_y)

    def interval(self, test_x):
        """
        Obtain the one-sigam confidence interval
        for a set of test points

        parameters
        ----------
        test_x : np.array
            locations where we want to test
        """
        test_x = np.array([test_x]).flatten()
        means, stds = [], []
        for row in test_x:
            S0 = self._exponential_cov(row, self.sample_x)
            v = cho_solve((self.lower_cholesky, True), S0)
            means.append(np.dot(S0, self.weighted_sample_y))
            stds.append(np.sqrt(self.width_scale ** 2 - np.dot(S0, v)))
        return means, stds

    def sample(self, test_x, samples=1):
        """
        Obtain function samples from the posterior

        parameters
        ----------
        test_x : np.array
            locations where we want to test

        samples : int
            Number of samples to take
        """
        S0 = self._exponential_cov(test_x, self.sample_x)
        # construct covariance matrix of sampled points.
        m = []
        for row in S0:
            m.append(cho_solve((self.lower_cholesky, True), row))
        cov = self._exponential_cov(test_x, test_x) - np.dot(S0, np.array(m).T)
        mean = np.dot(S0, self.weighted_sample_y)
        return np.random.multivariate_normal(mean, cov, samples)

def smooth(data, window):
    out0 = np.convolve(data, np.ones(window, dtype=int), 'valid') / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(data[:window - 1])[::2] / r
    stop = (np.cumsum(data[:-window:-1])[::2] / r)[::-1]
    return np.concatenate((  start , out0, stop  ))

# Insert data here.
data = pd.read_csv('./UAH/D1/20151110175712-16km-D1-NORMAL1-SECONDARY/RAW_GPS.txt',
                   sep='\s+', names=['time', 'speed'], usecols=[0, 1])
time = np.arange(10, 600, 0.1)
speed = np.interp(time, data['time'], data['speed'])
DURATION = 500
PREDICTION = 50
WIDTH_SCALE = 10
LENGTH_SCALE = 5
SAMPLES = 20
colors = cycle(['g', 'b', 'k', 'y', 'c', 'r', 'm'])
fig = plt.figure()
model = SimpleGP(WIDTH_SCALE, LENGTH_SCALE)
for i in range(10):
    ix = np.arange(i * DURATION, (i + 1) * DURATION + 1, PREDICTION)
    num = len(ix)
    sample_x = time[ix]
    sample_y = speed[ix] - np.mean(speed[ix])
    sample_s = 0.1 * np.random.rand(num) * np.ones_like(sample_x)
    m_speed = np.mean(speed[ix])
    model.fit(sample_x, sample_y, sample_s)

    test_x = np.arange(sample_x[0], sample_x[-1] + 5, .1)
    means, stds = model.interval(test_x)
    samples = model.sample(test_x, SAMPLES)

    # plots here.
    # ax = fig.add_subplot(511+i)
    plt.errorbar(test_x, means + m_speed, yerr=stds,
                 ecolor='g', linewidth=1.5,
                 elinewidth=0.5, alpha=0.75)

    for sample, c in zip(samples, colors):
        plt.plot(test_x, sample + m_speed, c, linewidth=2. * np.random.rand(), alpha=0.5)
    plt.plot([sample_x[0], sample_x[0]], [np.min(sample_y + m_speed), np.max(sample_y + m_speed)], c='r', ls='--')
    plt.plot([sample_x[-1], sample_x[-1]], [np.min(sample_y + m_speed), np.max(sample_y + m_speed)], c='r', ls='--')
    ix = np.arange(i * DURATION, (i + 1) * DURATION + PREDICTION, PREDICTION)
    real_x = time[ix]
    real_y = speed[ix]
    real_y = smooth(real_y, 3)
    plt.plot(real_x, real_y, c='b', linewidth=2)