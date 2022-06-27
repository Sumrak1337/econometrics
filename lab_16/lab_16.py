import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd

from sklearn.linear_model import LinearRegression
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Lab16:
    def __init__(self, stream):
        self.stream = stream
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

        self.t_crit = ss.t.ppf(1 - self.alpha_q, df=self.n-self.k-2)

        # for report
        self.lin_coefs = None
        self.pairs_lin_coefs = []
        self.pairs_intersect = []
        self.pairs_determ = []
        self.pairs_correlation = []
        self.pairs_avg_er = []
        self.intersect = None
        self.determ = None
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
        self.y_p = None
        self.interval = None

    def processing(self):
        x_list = [self.xs1, self.xs2]
        x_tags = ['xs1', 'xs2']
        for xs, tag in zip(x_list, x_tags):
            xs = xs.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(xs, self.ys)
            y_pred = reg.predict(xs)
            self.scatter_plot(xs, y_pred, tag)
            self.pairs_lin_coefs.append(reg.coef_)
            self.pairs_intersect.append(reg.intercept_)
            self.pairs_determ.append(reg.score(xs, self.ys))
            self.pairs_correlation.append(np.corrcoef(xs.reshape(1, -1), self.ys))
            self.pairs_avg_er.append(np.sum(np.abs(self.ys - y_pred) / np.abs(self.ys)) / len(self.ys))
            diffs = self.ys - y_pred
            self.residuals_plot(diffs, xs, tag)
            self.residuals_plot(diffs, range(len(xs)), tag=f'n_{tag}')

        self.multiregression()

    @staticmethod
    def residuals_plot(diffs, x, tag):
        plt.figure()
        plt.title(f'Диаграмма остатков ({tag})')
        plt.xlabel(f'{tag}')
        plt.ylabel('e')
        plt.plot(np.sort(x, axis=0), diffs)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'res_plot_{tag}')
        plt.close('all')

    def scatter_plot(self, xs, y, tag):
        plt.figure()
        plt.title(f'Корреляционное поле с линией парной регресии y({tag})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(xs, self.ys, alpha=0.5)
        plt.plot(xs, y, color=self.colors[1])
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'scatter_{tag}')
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

        reg = LinearRegression()
        reg.fit(x, y)
        self.lin_coefs = reg.coef_[0]
        self.intersect = reg.intercept_
        self.ys_pred = reg.predict(x).T[0]
        self.diffs = self.ys - self.ys_pred
        self.residuals_plot(self.diffs, range(len(x)), tag='xs')

        self.avg_er = np.sum(np.abs(self.ys - self.ys_pred) / np.abs(self.ys)) / len(self.ys)
        self.diffs_disp = np.sum((self.ys - self.ys_pred) ** 2) / (self.n - self.k - 1)
        self.determ = reg.score(self.xs, self.ys)
        ols = sm.OLS(y, x_wi)
        self.ols_results = ols.fit()
        self.f_stat = self.ols_results.f_pvalue
        self.m_a, self.m_b1, self.m_b2 = self.ols_results.bse

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
        self.y_p = x_p @ np.append([self.intersect], self.lin_coefs)
        self.interval = [self.y_p - m_y * self.t_crit, self.y_p + m_y * self.t_crit]

        self.report()

    @staticmethod
    def partial_coefs(r_y_x1, r_y_x2, r_x1_x2):
        r_y_x1_x2 = (r_y_x1 - r_y_x2 * r_x1_x2) / np.sqrt((1 - r_y_x2 ** 2) * (1 - r_x1_x2))
        return r_y_x1_x2

    def report(self):
        self.stream.write('Исходные данные:\n')
        self.stream.write(f'epsilons: \n{self.epsilons.round(3)}\n')
        self.stream.write(f'alpha: {self.alpha}\n')
        self.stream.write(f'beta1: {self.beta1}\n')
        self.stream.write(f'beta2: {self.beta2}\n')
        self.stream.write(f'xs1: \n{self.xs1.round(3)}\n')
        self.stream.write(f'xs2: \n{self.xs2.round(3)}\n')
        self.stream.write(f'ys = alpha + beta1 * xs1 + beta2 * xs2 + epsilon: \n{self.ys.round(3)}\n\n')

        self.stream.write('Парные линейные регрессии:\n')
        self.stream.write('Коэффициенты:\n')
        self.stream.write(f'ys/xs1 : {self.pairs_lin_coefs[0][0]:.3}, {self.pairs_intersect[0]:.3}\n')
        self.stream.write(f'ys/xs2: {self.pairs_lin_coefs[1][0]:.3}, {self.pairs_intersect[1]:.3}\n\n')

        self.stream.write('Коэффициент детерминации:\n')
        self.stream.write(f'xs1: {self.pairs_determ[0]:.3}\n')
        self.stream.write(f'xs2: {self.pairs_determ[1]:.3}\n\n')

        self.stream.write('Коэффициент корреляции:\n')
        self.stream.write(f'xs1: \n{self.pairs_correlation[0].round(3)}\n')
        self.stream.write(f'xs2: \n{self.pairs_correlation[1].round(3)}\n\n')

        self.stream.write('Средняя ошибка апроксимации:\n')
        self.stream.write(f'xs1: {self.pairs_avg_er[0]:.3}\n')
        self.stream.write(f'xs1: {self.pairs_avg_er[1]:.3}\n\n')

        self.stream.write('Множественная линейная регрессия\n\n')
        self.stream.write('Коэффициенты:\n')
        self.stream.write(f'ys/xs - b1, b2, a: {self.lin_coefs.round(3)}, {self.intersect.round(3)}\n\n')

        self.stream.write('Уравнение регрессии в развернутой форме:\n')
        self.stream.write(f'y = {self.intersect[0]:.3} + ({self.lin_coefs[0]:.3}) * x1 + ({self.lin_coefs[1]:.3}) * x2\n\n')

        self.stream.write('Коэффициент детерминации:\n')
        self.stream.write(f'{self.determ:.3}\n\n')

        self.stream.write('Значения F-статистики\n')
        self.stream.write(f'{self.ols_results.summary()}\n\n')

        self.stream.write('Средняя ошибка апроксимации:\n')
        self.stream.write(f'{self.avg_er:.3}\n\n')

        self.stream.write('Оценка для дисперсии остатков\n')
        self.stream.write(f'{self.diffs_disp:.3}\n\n')

        self.stream.write('Частные коэффициенты корреляции\n')
        self.stream.write(f'r_yx1|x2: {self.r12:.3}\n')
        self.stream.write(f'r_yx2|x1: {self.r21:.3}\n\n')

        self.stream.write('Стандартные ошибки коэффициентов регрессии\n')
        self.stream.write(f'a: {self.m_a:.3}\n')
        self.stream.write(f'b1: {self.m_b1:.3}\n')
        self.stream.write(f'b2: {self.m_b2:.3}\n\n')

        std_er_a, std_er_b1, std_er_b2 = self.ols_results.bse
        t_a = self.intersect[0] / std_er_a
        t_b1 = self.lin_coefs[0] / std_er_b1
        t_b2 = self.lin_coefs[1] / std_er_b2

        if t_a > self.t_crit:
            self.stream.write(f'Коэффициент a статистически значим, \nтак как t_a > t_crit ({t_a:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент a статистически не значим, \nтак как t_a <= t_crit ({t_a:.3} <= {self.t_crit:.3})\n')

        if t_b1 > self.t_crit:
            self.stream.write(f'Коэффициент b1 статистически значим, \nтак как t_b1 > t_crit ({t_b1:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент b1 статистически не значим, \nтак как t_b1 <= t_crit ({t_b1:.3} <= {self.t_crit:.3})\n')

        if t_b2 > self.t_crit:
            self.stream.write(f'Коэффициент b2 статистически значим, \nтак как t_b2 > t_crit ({t_b2:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент b2 статистически не значим, \nтак как t_b2 <= t_crit ({t_b2:.3} <= {self.t_crit:.3})\n\n')

        self.stream.write('Точечный прогноз\n')
        self.stream.write(f'Значение: {self.y_p:.3}\n')
        self.stream.write(f'Интервальный прогноз: {np.array(self.interval).round(3)}\n')


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
Lab16(file).processing()
file.close()
