import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss

from data.lab_data import LabData
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'
# TODO: add mod quantiles


class Lab05:

    def __init__(self, main_data: dict, stream):
        self.stream = stream

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Исходные данные\n')
        for k, v in main_data.items():
            data.write(f'{k}: {v}\n')
        data.close()

        self.xs = np.array([value[0] for value in main_data.values()])
        self.ys = np.array([value[1] for value in main_data.values()])

        self.xs_diffs = np.array([self.xs[i] - self.xs[i - 1] for i in range(1, len(self.xs))]).round(decimals=2)
        self.ys_diffs = np.array([self.ys[i] - self.ys[i - 1] for i in range(1, len(self.ys))]).round(decimals=2)
        self.n = len(self.xs_diffs)
        self.m = len(self.ys_diffs)
        self.xs_mean = np.mean(self.xs_diffs)
        self.ys_mean = np.mean(self.ys_diffs)
        self.xs_disp = np.var(self.xs_diffs)
        self.ys_disp = np.var(self.ys_diffs)

        self.alpha = 0.05
        self.quantile = 1.96
        self.mod_quantiles = {0.15: [0.775, 0.091],
                              0.10: [0.819, 0.104],
                              0.05: [0.895, 0.126],
                              0.025: [0.955, 0.148],
                              0.01: [1.035, 0.178]}

    def processing(self):
        self.student_known_variances()
        self.student_unknown_variances()
        self.fisher_snedekor()
        self.stream.write('Экспорт\n')
        self.kolmogorov(self.xs_diffs, 'Export')
        self.stream.write('Импорт\n')
        self.kolmogorov(self.ys_diffs, 'Import')

    def student_known_variances(self):
        phi_1 = (np.mean(self.xs_diffs) - np.mean(self.ys_diffs)) / np.sqrt(self.xs_disp / self.n + self.ys_disp / self.m)

        # report
        self.stream.write('Критерий Стьюдента, известные дисперсии\n')

        if np.abs(phi_1) > self.quantile:
            self.stream.write(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, \nтак как значение phi1 = |{phi_1:.4}| принадлежит интервалу ({self.quantile}; +inf)\n\n')
        else:
            self.stream.write(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, \nтак как значение phi1 = |{phi_1:.4}| не принадлежит интервалу ({self.quantile}; +inf)\n\n')

    def student_unknown_variances(self):
        s = np.sqrt((self.n * self.xs_disp + self.m * self.ys_disp) / (self.n + self.m - 2))
        phi_2 = (self.xs_mean - self.ys_mean) / (s * np.sqrt(1 / self.n + 1 / self.m))

        # report
        self.stream.write('Критерий Стьюдента, неизвестные дисперсии\n')

        if np.abs(phi_2) > self.quantile:
            self.stream.write(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, \nтак как значение phi2 = |{phi_2:.4}| принадлежит интервалу ({self.quantile}; +inf)\n\n')
        else:
            self.stream.write(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, \nтак как значение phi2 = |{phi_2:.4}| не принадлежит интервалу ({self.quantile}; +inf)\n\n')

    def fisher_snedekor(self):
        f = np.var(self.xs_diffs, ddof=1) / np.var(self.ys_diffs, ddof=1)
        u1 = 0.4994
        u2 = 2.0023

        # report
        self.stream.write('Критерий Фишера-Снедекора, неизвестные дисперсии\n')

        if (0 <= f < u1) or (u2 < f < np.inf):
            self.stream.write(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, \nтак как значение F = {f:.4} принадлежит интервалу [0; {u1}) U ({u2}; +inf)\n\n')
        else:
            self.stream.write(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, \nтак как значение F = {f:.4} не принадлежит интервалу [0; {u1}) U ({u2}; +inf)\n\n')

    def kolmogorov(self, diffs, label):
        min_diffs = np.min(diffs)
        max_diffs = np.max(diffs)
        cum_freq = ss.cumfreq(diffs, numbins=self.n - 1, defaultreallimits=(min_diffs, max_diffs))
        cum_freq = np.append([1], cum_freq[0])
        freq = cum_freq / len(diffs)

        intervals = np.linspace(min_diffs, max_diffs, self.n)
        edf = sm.distributions.StepFunction(intervals, freq, side='right')

        plt.figure()
        plt.title(f'{label}')
        plt.xlabel('Вариационный ряд')
        plt.ylabel('F(x)')
        plt.step(intervals, edf(intervals))
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'{label}')
        plt.close('all')

        left_value = np.max(np.array([i / self.n - edf.y[i] for i in range(1, self.n + 1)]))
        right_value = np.max(np.array([edf.y[i] - (i - 1) / self.n for i in range(1, self.n + 1)]))
        dn = max(left_value, right_value)
        dn = dn * (np.sqrt(self.n) - 0.01 + 0.85 / np.sqrt(self.n))

        # report
        self.stream.write('Критерий Колмогорова\n')
        d_star = self.mod_quantiles[self.alpha][0]

        if dn <= d_star:
            self.stream.write(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, \nтак как значение {dn:.6} не попадает в интервал ({d_star}; +inf)\n\n')
        else:
            self.stream.write(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, \nтак как значение {dn:.6} попадает в интервал ({d_star}; +inf)\n\n')


file = open(RESULT_ROOT / 'file_test.txt', 'w', encoding='utf-8')
dict_data = LabData().lab_05_data
Lab05(dict_data, file).processing()
file.close()
