import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from data.test_data import TestData02

from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Test02:
    def __init__(self, main_data, stream):
        self.stream = stream
        self.init = main_data

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Исходные данные\n')
        data.write(f'{self.init}\n')
        data.close()

        self.x1 = self.init[:, 0]
        self.x2 = self.init[:, 1]
        self.y = self.init[:, -1]

        n, k = self.init[:, :2].shape
        self.alpha = 0.05
        self.t_crit = ss.t.ppf(1 - self.alpha, df=n-k-2)

        self.diffs = []

    def processing(self):
        self.paired_regressions(self.y, self.x1.reshape(-1, 1), 'x1')
        self.paired_regressions(self.y, self.x2.reshape(-1, 1), 'x2')
        self.multiregression()
        self.jarque_baire()

    def paired_regressions(self, y, x, tag):
        self.stream.write(f'Парная регрессия y ~ {tag}\n\n')
        reg = LinearRegression()
        reg.fit(x, y)

        coef = reg.coef_[0]
        intersect = reg.intercept_

        self.stream.write('Коэффициенты линейной регрессии:\n')
        self.stream.write(f'a: {intersect:.3f}\n')
        self.stream.write(f'b: {coef:.3f}\n\n')

        plt.figure()
        plt.title(f'Корреляционное поле с парной линейной регрессией Y ~ {tag}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, reg.predict(x), color=list(mcolors.TABLEAU_COLORS)[1])
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'lin_reg_y_{tag}')
        plt.close('all')

        diffs = y - reg.predict(x)
        self.diffs.append(diffs)
        plt.figure()
        plt.title(f'Диаграмма остатков y ~ {tag}')
        plt.xlabel(f'{tag}')
        plt.ylabel('Y')
        plt.plot(np.sort(x, axis=0), diffs)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'res_plot_{tag}')
        plt.close('all')

        plt.figure()
        plt.title(f'Диаграмма остатков y ~ n_{tag}')
        plt.xlabel(f'n_{tag}')
        plt.ylabel('Y')
        plt.plot(range(len(x)), diffs)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'res_plot_n_{tag}')
        plt.close('all')

        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        n, k = x.shape
        p = len(x.columns) + 1
        x_wi = np.empty(shape=(n, p))
        x_wi[:, 0] = 1
        x_wi[:, 1:p] = x.values

        ols = sm.OLS(y, x_wi)
        ols_results = ols.fit()
        std_er_a, std_er_b = ols_results.bse

        t_a = intersect / std_er_a
        t_b = coef / std_er_b

        if t_a > self.t_crit:
            self.stream.write(f'Коэффициент a статистически значим на уровне значимости {self.alpha}, \nтак как t_a > t_crit ({t_a:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент a статистически не значим на уровне значимости {self.alpha}, \nтак как t_a <= t_crit ({t_a:.3} <= {self.t_crit:.3})\n')

        if t_b > self.t_crit:
            self.stream.write(f'Коэффициент b статистически значим на уровне значимости {self.alpha}, \nтак как t_b > t_crit ({t_b:.3} > {self.t_crit:.3})\n\n')
        else:
            self.stream.write(f'Коэффициент b статистически не значим на уровне значимости {self.alpha}, \nтак как t_b <= t_crit ({t_b:.3} <= {self.t_crit:.3})\n\n')

        matrix = np.linalg.inv(x_wi.T @ x_wi)
        xp = np.array([1, np.mean(x.values) * 2.75])
        m_y = xp @ matrix @ xp.T
        yp = xp @ np.append([intersect], coef)
        interval = [yp - m_y * self.t_crit, yp + m_y * self.t_crit]

        self.stream.write('Точечный прогноз:\n')
        self.stream.write(f'yp: {yp:.3f}\n')
        self.stream.write('Интервальный прогноз:\n')
        self.stream.write(f'{np.array(interval).round(3)}\n\n')

    def multiregression(self):
        self.stream.write('Множественная регрессия\n\n')
        reg = LinearRegression()
        x = pd.DataFrame(self.init[:, :2])
        y = pd.DataFrame(self.y)
        reg.fit(x, y)

        coefs = reg.coef_[0]
        intercect = reg.intercept_[0]

        self.stream.write('Коэффициенты линейной регрессии:\n')
        self.stream.write(f'a: {np.array(intercect).round(3)}\n')
        self.stream.write(f'b: {np.array(coefs).round(3)}\n\n')

        n = len(x)
        p = len(x.columns) + 1
        x_wi = np.empty(shape=(n, p))
        x_wi[:, 0] = 1
        x_wi[:, 1:p] = x.values

        ols = sm.OLS(y, x_wi)
        ols_results = ols.fit()
        std_er_a, std_er_b1, std_er_b2 = ols_results.bse

        t_a = intercect / std_er_a
        t_b1 = coefs[0] / std_er_b1
        t_b2 = coefs[1] / std_er_b2

        if t_a > self.t_crit:
            self.stream.write(f'Коэффициент a статистически значим на уровне значимости {self.alpha}, \nтак как t_a > t_crit ({t_a:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент a статистически не значим на уровне значимости {self.alpha}, \nтак как t_a <= t_crit ({t_a:.3} <= {self.t_crit:.3})\n')

        if t_b1 > self.t_crit:
            self.stream.write(f'Коэффициент b1 статистически значим на уровне значимости {self.alpha}, \nтак как t_b1 > t_crit ({t_b1:.3} > {self.t_crit:.3})\n')
        else:
            self.stream.write(f'Коэффициент b1 статистически не значим на уровне значимости {self.alpha}, \nтак как t_b1 <= t_crit ({t_b1:.3} <= {self.t_crit:.3})\n')

        if t_b2 > self.t_crit:
            self.stream.write(f'Коэффициент b2 статистически значим на уровне значимости {self.alpha}, \nтак как t_b2 > t_crit ({t_b2:.3} > {self.t_crit:.3})\n\n')
        else:
            self.stream.write(f'Коэффициент b2 статистически не значим на уровне значимости {self.alpha}, \nтак как t_b2 <= t_crit ({t_b2:.3} <= {self.t_crit:.3})\n\n')

        diffs = self.y - reg.predict(x).reshape(1, -1)[0]
        self.diffs.append(diffs)

        plt.figure()
        plt.title('Диаграмма остатков на номера наблюдений')
        plt.xlabel('n')
        plt.ylabel('deltaY')
        plt.plot(range(len(self.y)), diffs)
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / 'res_plot_gen')
        plt.close('all')

        xp = np.array([1, np.mean(self.x1) * 2.75, np.mean(self.x2) * 2.75])
        yp = xp @ np.append(intercect, coefs)

        matrix = np.linalg.inv(x_wi.T @ x_wi)
        m_y = xp @ matrix @ xp.T
        interval = [yp - m_y * self.t_crit, yp + m_y * self.t_crit]

        self.stream.write('Точечный прогноз:\n')
        self.stream.write(f'yp: {yp:.3f}\n')
        self.stream.write('Интервальный прогноз:\n')
        self.stream.write(f'{np.array(interval).round(3)}\n\n')

    def jarque_baire(self):
        tags = ['y1', 'y2', 'y3']
        for diffs, tags in zip(self.diffs, tags):
            s = ss.skew(diffs)
            k = ss.kurtosis(diffs)
            n = len(diffs)
            jb = n * (s ** 2 + (k - 3) ** 2 / 4) / 6
            chi = ss.chi2.ppf(self.alpha, df=2)

            self.stream.write('Критерий Жака-Бера\n')
            if jb > chi:
                self.stream.write(f'Гипотеза H0 о нормальности распределения остатков \nотвергается на уровне значимости {self.alpha}, \n'
                                  f'так как статистика Жака-Бера больше критического значения распределения хи-квадрат \n'
                                  f'({jb:.3} > {chi:.3})\n\n')
            else:
                self.stream.write(f'Гипотеза H0 о нормальности распределения остатков \nпринимается на уровне значимости {self.alpha}, \n'
                                  f'так как статистика Жака-Бера не превышает критического значения распределения хи-квадрат \n'
                                  f'({jb:.3} <= {chi:.3})\n\n')


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
test_data = TestData02.test_11()
Test02(test_data, file).processing()
file.close()
