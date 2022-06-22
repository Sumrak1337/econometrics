import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
# TODO: add graphs labels
# TODO: create nice report


class Lab16:
    def __init__(self):
        self.a = 0
        self.sigma = 2
        self.alpha_q = 0.05
        self.k = 2
        self.n = 30
        self.colors = list(mcolors.TABLEAU_COLORS)

        self.rs = np.random.RandomState(42)
        self.epsilons = ss.norm.ppf(self.rs.random(self.n), loc=self.a, scale=self.sigma)
        self.xs1 = ss.norm.ppf(self.rs.random(self.n), loc=self.rs.randint(0, 100), scale=self.rs.randint(1, 100))
        self.xs2 = ss.norm.ppf(self.rs.random(self.n), loc=self.rs.randint(0, 100), scale=self.rs.randint(1, 100))
        self.xs = pd.concat((pd.DataFrame(self.xs1), pd.DataFrame(self.xs2)), axis=1, ignore_index=True)

        self.alpha = round(self.rs.random(), 3)
        self.beta1 = round(self.rs.random(), 3)
        self.beta2 = round(self.rs.random(), 3)

        self.ys = self.alpha + self.beta1 * self.xs1 + self.beta2 * self.xs2 + self.epsilons

        self.t_crit = ss.t.ppf(1 - self.alpha_q, df=self.n-self.k-2)  # ?

        # for report
        self.lin_coefs = None
        self.intersect = None
        self.determ = None
        self.correlation = None
        self.ys_pred = None
        self.diffs = None
        self.avg_er = None
        self.diffs_disp = None
        self.ols_results = None
        self.f_stat = None
        self.m_a = None
        self.m_b1 = None
        self.m_b2 = None
        self.r12 = None
        self.r21 = None
        self.xp = None
        self.yp = None

    def processing(self):
        x_list = [self.xs1, self.xs2]
        for xs in x_list:
            xs = xs.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(xs, self.ys)
            y_pred = reg.predict(xs)
            self.scatter_plot(xs, y_pred)
            lin_coefs = reg.coef_
            intersect = reg.intercept_
            determ = reg.score(xs, self.ys)
            correlation = np.corrcoef(xs.reshape(1, -1), self.ys)
            avg_er = np.sum(np.abs(self.ys - y_pred) / np.abs(self.ys)) / len(self.ys)
            diffs = self.ys - y_pred
            self.residuals_plot(diffs, xs)
            self.residuals_plot(diffs, range(len(xs)))

        self.multiregression()

    @staticmethod
    def residuals_plot(diffs, x):
        plt.figure()
        plt.plot(np.sort(x, axis=0), diffs)
        plt.show()
        plt.close('all')

    def scatter_plot(self, xs, y):
        plt.figure()
        plt.scatter(xs, self.ys, alpha=0.5)
        plt.plot(xs, y, color=self.colors[1])
        plt.show()
        plt.close('all')

    def multiregression(self):
        x = self.xs
        y = pd.DataFrame(self.ys)
        n = len(x)
        p = len(x.columns) + 1
        x_wi = np.empty(shape=(n, p))
        x_wi[:, 0] = 1
        x_wi[:, 1:p] = x.values

        coefs = (np.linalg.inv(x_wi.T @ x_wi) @ x_wi.T @ y.values).T
        print('Уравнение регрессии в развернутой форме:')
        a, b1, b2 = zip(*coefs)
        print(f'y = {a[0]:.3} + ({b1[0]:.3}) * x1 + ({b2[0]:.3}) * x2')

        reg = LinearRegression()
        reg.fit(x, y)
        self.lin_coefs = reg.coef_[0]
        self.intersect = reg.intercept_
        self.ys_pred = reg.predict(x).T[0]
        self.diffs = self.ys - self.ys_pred
        self.residuals_plot(self.diffs, range(len(x)))

        self.avg_er = np.sum(np.abs(self.ys - self.ys_pred) / np.abs(self.ys)) / len(self.ys)
        self.diffs_disp = np.sum((self.ys - self.ys_pred) ** 2) / (self.n - self.k - 1)
        self.determ = reg.score(self.xs, self.ys)
        ols = sm.OLS(y, x_wi)
        ols_results = ols.fit()
        self.f_stat = ols_results.f_pvalue
        self.m_a, self.m_b1, self.m_b2 = ols_results.bse
        print(ols_results.summary())

        r_x1_x2 = np.corrcoef(self.xs1, self.xs2)[0, 1]
        r_y_x1 = np.corrcoef(self.ys, self.xs1)[0, 1]
        r_y_x2 = np.corrcoef(self.ys, self.xs2)[0, 1]
        self.r12 = self.partial_coefs(r_y_x1, r_y_x2, r_x1_x2)
        self.r21 = self.partial_coefs(r_y_x2, r_y_x1, r_x1_x2)

        self.xp = (self.xs + 3 * np.mean(self.xs)).to_numpy()
        self.yp = self.intersect + self.lin_coefs @ self.xp.T

        matrix = np.linalg.inv(x_wi.T @ x_wi)
        x_p = np.array([1, np.mean(self.xs1) * 3, np.mean(self.xs2) * 3])
        m_y = x_p @ matrix @ x_p.T
        y_p = x_p @ np.append([self.intersect], self.lin_coefs)
        interval = [y_p - m_y * self.t_crit, y_p + m_y * self.t_crit]

        self.report()

    @staticmethod
    def partial_coefs(r_y_x1, r_y_x2, r_x1_x2):
        r_y_x1_x2 = (r_y_x1 - r_y_x2 * r_x1_x2) / np.sqrt((1 - r_y_x2 ** 2) * (1 - r_x1_x2))
        return r_y_x1_x2

    def report(self):
        ...


Lab16().processing()
