import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Lab14:
    def __init__(self, stream):
        self.stream = stream
        self.a = 0
        self.sigma = 2
        self.alpha_q = 0.05
        self.colors = list(mcolors.TABLEAU_COLORS)

        self.rs = np.random.RandomState(0)
        self.epsilons = ss.norm.ppf(self.rs.random(50), loc=self.a, scale=self.sigma)
        self.xs = ss.norm.ppf(self.rs.random(50), loc=self.rs.randint(0, 100), scale=self.rs.randint(0, 100)).astype(float)

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
        self.yp = None

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
        plt.title('Диаграмма корреляционного поля')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(self.xs, self.ys, alpha=0.5)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / 'cor_field')
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
        plt.title('График остатков по отношению к X')
        plt.xlabel('X')
        plt.ylabel('e')
        plt.plot(np.sort(self.xs), self.diffs)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / 'X_e')

        plt.figure()
        plt.title('График остатков по отношению к номеру наблюдения')
        plt.xlabel('n')
        plt.ylabel('e')
        plt.plot(range(len(self.xs)), self.diffs)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / 'n_e')
        plt.close('all')

    def scatter_plot_with_regression(self):
        plt.figure()
        plt.title('Корреляционное поле с линией регрессии')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(self.xs, self.ys, alpha=0.5)
        plt.plot(self.xs, self.ys_pred, color=self.colors[1])
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / 'cor_field_with_reg')
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
        plt.title('Точечный прогноз')
        plt.xlabel('X')
        plt.ylabel('Y')
        self.xp = self.xs + 3 * np.mean(self.xs)
        y = self.intersect + self.lin_coefs * self.xp
        plt.scatter(self.xs, self.ys, alpha=0.5)
        plt.scatter(self.xp, y, alpha=0.5)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / 'predict')
        plt.close('all')

    def standard_y_error(self):
        rmse = mse(self.ys, self.ys_pred)
        xp = np.mean(self.xs) * 3
        self.yp = self.intersect + self.lin_coefs * xp
        self.m_y = rmse * np.sqrt(1 + 1 / len(self.xs) + (xp - np.mean(self.xs)) ** 2 / np.sum((self.xs - np.mean(self.xs)) ** 2))
        self.m_y_left = self.yp - self.m_y * self.t_crit
        self.m_y_right = self.yp + self.m_y * self.t_crit

    def report(self):
        self.stream.write('Исходные данные:\n')
        self.stream.write(f'epsilons: \n{self.epsilons.round(3)}\n')
        self.stream.write(f'alpha: {self.alpha}\n')
        self.stream.write(f'beta: {self.beta}\n')
        self.stream.write(f'xs: \n{self.xs.round(3)}\n')
        self.stream.write(f'ys = alpha + beta * xs + epsilons: \n{self.ys.round(3)}\n\n')

        self.stream.write(f'Найденное уравнение линейной регрессии:\n')
        self.stream.write(f'y = {self.intersect:.3} + ({self.lin_coefs[0]:.3})*X\n\n')

        self.stream.write('Коэффициент корреляции:\n')
        self.stream.write(f'{self.correlation:.3}\n\n')

        self.stream.write('Коэффициент детерминации:\n')
        self.stream.write(f'{self.determ:.3}\n\n')

        self.stream.write('Средняя ошибка апроксимации:\n')
        self.stream.write(f'{self.avg_er:.3}\n\n')

        self.stream.write('Результаты F-статистики:\n')
        self.stream.write(f'{self.ols_results.summary()}\n\n')

        self.stream.write('Значения t-статистик:\n')
        self.stream.write(f'для коэффициента корреляции: {self.t_corr:.3}\n')
        self.stream.write(f'для alpha: {self.t_a:.3}\n')
        self.stream.write(f'для beta: {self.t_b:.3}\n')
        if self.t_corr > self.t_crit:
            self.stream.write(f'Коэффициент корреляции статистически значим, \nтак как t_corr > t_crit ({self.t_corr:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент корреляции статистически не значим, \nтак как t_corr <= t_crit ({self.t_corr:.3} <= {self.t_crit:.3})\n')

        if self.t_a > self.t_crit:
            self.stream.write(f'Коэффициент а статистически значим, \nтак как t_а > t_crit ({self.t_a:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент a статистически не значим, \nтак как t_a <= t_crit ({self.t_a:.3} <= {self.t_crit:.3})\n')

        if self.t_b > self.t_crit:
            self.stream.write(f'Коэффициент b статистически значим, \nтак как t_b > t_crit ({self.t_b:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент b статистически не значим, \nтак как t_b <= t_crit ({self.t_b:.3} <= {self.t_crit:.3})\n\n')

        self.stream.write('Доверительные интервалы:\n')
        self.stream.write(f'для alpha: {self.conf_int_a.round(3)}\n')
        self.stream.write(f'для beta: {self.conf_int_b.round(3)}\n\n')

        self.stream.write('Значение точечного прогноза:\n')
        self.stream.write(f'{self.yp[0]:.3}\n\n')

        self.stream.write('Ошибка точечного прогноза:\n')
        self.stream.write(f'{self.m_y:.3}\n\n')

        self.stream.write('Доверительный интервал для прогноза:\n')
        self.stream.write(f'({self.m_y_left[0]:.3}; {self.m_y_right[0]:.3})\n')


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
Lab14(file).processing()
file.close()
