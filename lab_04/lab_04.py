import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss

from data.lab_data import LabData
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'

# TODO: also add demography stats
# TODO: add critical values


class Lab04:
    def __init__(self, main_data: dict, stream):
        self.stream = stream
        self.xs = np.array([value[0] for value in main_data.values()])
        self.ys = np.array([value[1] for value in main_data.values()])

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Исходные данные\n')
        for k, x, y in zip(list(main_data), self.xs, self.ys):
            data.write(f'{k}: {x}, {y}\n')
        data.close()

        self.xs_diffs = self.diffs(self.xs)
        self.ys_diffs = self.diffs(self.ys)

        table = open(RESULT_ROOT / 'table.txt', 'w', encoding='utf-8')
        table.write('Первые разности\n')
        for x, y in zip(self.xs_diffs, self.ys_diffs):
            table.write(f'{x}, {y}\n')
        table.close()

        self.n = len(self.xs_diffs)
        self.m = len(self.ys_diffs)

        self.omega = 96.
        self.alphas = [0.05]
        self.mod_quantiles = {0.005: 1.73,
                              0.01: 1.63,
                              0.025: 1.48,
                              0.05: 1.36,
                              0.1: 1.22,
                              0.15: 1.14,
                              0.2: 1.07,
                              0.25: 1.02}

    def processing(self):
        self.kolmogorov_smirnov()
        self.wilcoxon()

    def kolmogorov_smirnov(self):
        f_n = self.edf(self.xs_diffs, 'X')
        g_m = self.edf(self.ys_diffs, 'Y')

        d_mn1 = np.max(np.array([r / self.n - f_n.y[r] for r in range(1, self.n + 1)]))
        d_mn2 = np.max(np.array([f_n.y[r] - (r - 1) / self.n for r in range(1, self.n + 1)]))

        d_mn = max(d_mn1, d_mn2)

        d_mn = np.sqrt(self.n ** 2 / (2 * self.n)) * d_mn

        # report
        self.stream.write('Критерий Колмогорова-Смирнова\n')

        for alpha in self.alphas:
            if d_mn > self.mod_quantiles[alpha]:
                self.stream.write(f'Гипотеза H0 отвергается на уровне значимости {alpha}, \nтак значение {d_mn:.4} принадлежитлежит внутри интервала ({self.mod_quantiles[alpha]}; +inf)\n\n')
            else:
                self.stream.write(f'Гипотеза H0 принимается на уровне значимости {alpha}, \nтак значение {d_mn:.4} не принадлежит интервалу ({self.mod_quantiles[alpha]}; +inf)\n\n')

    def wilcoxon(self):
        general = np.append(self.xs_diffs, self.ys_diffs)
        general.sort()
        ranges = self.ranges_processing(general)
        indices = np.isin(general, self.ys_diffs)
        w = np.sum(ranges[indices])

        # report
        self.stream.write('Критерий Вилкоксона\n')
        for alpha in self.alphas:
            u1 = self.m * (self.m + 1) / 2
            u3 = self.m * (self.n + self.m + 1) - self.omega
            u4 = self.m * self.n + self.m * (self.m + 1) / 2
            if (u1 <= w <= self.omega) or (u3 <= w <= u4):
                self.stream.write(f'Гипотеза H0 отвергается на уровне значимости {alpha}, \nтак как значение w = {w:4} попадает \nв критическую область [{u1}; {self.omega}] U [{u3}; {u4}]\n\n')
            else:
                self.stream.write(f'Гипотеза H0 принимается на уровне значимости {alpha}, \nтак как значение w = {w:4} не принадлежит \nкритической области [{u1}; {self.omega}] U [{u3}; {u4}]\n\n')

    @staticmethod
    def edf(diffs: np.array, label):
        n = len(diffs)
        min_diffs = np.min(diffs)
        max_diffs = np.max(diffs)

        cum_freq = ss.cumfreq(diffs, numbins=n-1, defaultreallimits=(min_diffs, max_diffs))
        cum_freq = np.append([1], cum_freq[0])
        freq = cum_freq / n

        intervals = np.linspace(min_diffs, max_diffs, n)

        ecdf = sm.distributions.StepFunction(intervals, freq, side='right')

        plt.figure()
        plt.title(f'Эмпирическая функция {label}')
        plt.xlabel('Вариационный ряд')
        plt.ylabel('F(x)')
        plt.step(intervals, ecdf(intervals))
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'ecdf_{label}')
        plt.close('all')

        return ecdf

    @staticmethod
    def diffs(data: np.array):
        d = np.array([data[i] - data[i - 1] for i in range(1, len(data))])
        return d.round(decimals=2)

    @staticmethod
    def ranges_processing(general):
        unique, indices, counts = np.unique(general, return_index=True, return_counts=True)
        res = np.array([])
        for cnt, idx in zip(counts, indices):
            if cnt > 1:
                value = (2 * idx + cnt + 1) / 2
            else:
                value = idx + 1
            res = np.append(res, [value] * cnt)

        return res


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
dict_data = LabData().lab_04_data
Lab04(dict_data, file).processing()
file.close()
