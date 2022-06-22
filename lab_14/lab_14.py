import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
# TODO: add graphs labels
# TODO: create nice report


class Lab14:
    def __init__(self):
        self.a = 0
        self.sigma = 2
        self.alpha_q = 0.05
        self.colors = list(mcolors.TABLEAU_COLORS)

        self.rs = np.random.RandomState(0)
        self.epsilons = ss.norm.ppf(self.rs.random(50), loc=self.a, scale=self.sigma)
        self.xs = ss.norm.ppf(self.rs.random(50), loc=self.rs.randint(0, 100), scale=self.rs.randint(0, 100))

        self.alpha = round(self.rs.random(), 3)
        self.beta = round(self.rs.random(), 3)

        self.ys = self.alpha + self.beta * self.xs + self.epsilons

        self.t_crit = ss.t.ppf(1 - self.alpha_q, df=len(self.xs)-2)

        # for report
        self.lin_coefs = None
        self.intersect = None
        self.determ = None
        self.correlation = None
        self.ys_pred = None
        self.diffs = None
        self.avg_er = None
        self.ols_results = None
        self.t_corr = None
        self.t_a = None
        self.t_b = None
        self.conf_int_a = None
        self.conf_int_b = None
        self.xp = None
        self.m_y = None
        self.m_y_left = None
        self.m_y_right = None

    def processing(self):
        self.scatter_plot()
        self.calculation()
        self.residuals_plots()
        self.scatter_plot_with_regression()
        self.t_stat()
        self.plot_200_predict()
        self.standard_y_error()
        self.report()

    def scatter_plot(self):
        plt.figure()
        plt.scatter(self.xs, self.ys, alpha=0.5)
        plt.show()
        plt.close('all')

    def calculation(self):
        reg = LinearRegression()
        x = self.xs.reshape(-1, 1)
        reg.fit(x, self.ys)
        self.lin_coefs = reg.coef_
        self.intersect = reg.intercept_
        self.determ = reg.score(x, self.ys)
        self.correlation = np.corrcoef(self.xs, self.ys)[0, 1]
        self.ys_pred = reg.predict(x)
        self.diffs = self.ys - self.ys_pred
        self.avg_er = np.sum(np.abs(self.ys - self.ys_pred) / np.abs(self.ys)) / len(self.ys)

    def residuals_plots(self):
        plt.figure()
        plt.plot(np.sort(self.xs), self.diffs)
        plt.show()

        plt.figure()
        plt.plot(range(len(self.xs)), self.diffs)
        plt.show()
        plt.close('all')

    def scatter_plot_with_regression(self):
        plt.figure()
        plt.scatter(self.xs, self.ys, alpha=0.5)
        plt.plot(self.xs, self.ys_pred, color=self.colors[1])
        plt.show()
        plt.close('all')

    def t_stat(self):
        self.t_corr = self.correlation * np.sqrt(1 - self.correlation ** 2) / np.sqrt(len(self.xs) - 2)

        x = pd.DataFrame(self.xs)
        y = pd.DataFrame(self.ys)
        n = len(x)
        p = len(x.columns) + 1
        x_wi = np.empty(shape=(n, p))
        x_wi[:, 0] = 1
        x_wi[:, 1:p] = x.values
        ols = sm.OLS(y.values, x_wi)

        self.ols_results = ols.fit()
        std_er_a, std_er_b = self.ols_results.bse

        self.t_b = self.lin_coefs[0] / std_er_b
        self.t_a = self.intersect / std_er_a

        self.conf_int_a, self.conf_int_b = self.ols_results.conf_int(alpha=self.alpha_q)

    def plot_200_predict(self):
        plt.figure()
        self.xp = self.xs + 3 * np.mean(self.xs)
        y = self.intersect + self.lin_coefs * self.xp
        plt.scatter(self.xs, self.ys, alpha=0.5)
        plt.scatter(self.xp, y, alpha=0.5)
        plt.show()
        plt.close('all')

    def standard_y_error(self):
        rmse = mse(self.ys, self.ys_pred)
        xp = np.mean(self.xs) * 3
        yp = self.intersect + self.lin_coefs * xp
        self.m_y = rmse * np.sqrt(1 + 1 / len(self.xs) + (xp - np.mean(self.xs)) ** 2 / np.sum((self.xs - np.mean(self.xs)) ** 2))
        self.m_y_left = yp - self.m_y * self.t_crit
        self.m_y_right = yp + self.m_y * self.t_crit

    def report(self):
        ...


Lab14().processing()
