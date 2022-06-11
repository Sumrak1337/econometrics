import numpy as np
import statsmodels.api as sm
import scipy.stats as ss

from data.lab_data import LabData

# TODO: also add demography stats


class Lab04:

    def __init__(self, main_data: dict):
        self.xs = np.array([value[0] for value in main_data.values()])
        self.ys = np.array([value[1] for value in main_data.values()])

        self.xs_diffs = self.diffs(self.xs)
        self.ys_diffs = self.diffs(self.ys)

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
        f_n = self.edf(self.xs_diffs)
        g_m = self.edf(self.ys_diffs)

        d_mn1 = np.max(np.array([r / self.n - f_n.y[r] for r in range(1, self.n + 1)]))
        d_mn2 = np.max(np.array([f_n.y[r] - (r - 1) / self.n for r in range(1, self.n + 1)]))

        d_mn = max(d_mn1, d_mn2)

        d_mn = np.sqrt(self.n ** 2 / (2 * self.n)) * d_mn

        # report
        print('Критерий Колмогорова-Смирнова')

        for alpha in self.alphas:
            if d_mn > self.mod_quantiles[alpha]:
                print(f'Гипотеза H0 отвергается на уровне значимости {alpha}, так значение {d_mn:.8} принадлежитлежит внутри интервала ({self.mod_quantiles[alpha]}; +inf)')
            else:
                print(f'Гипотеза H0 принимается на уровне значимости {alpha}, так значение {d_mn:.8} не принадлежит интервалу ({self.mod_quantiles[alpha]}; +inf)')
            print()

    def wilcoxon(self):
        general = np.append(self.xs_diffs, self.ys_diffs)
        general.sort()
        ranges = self.ranges_processing(general)
        indices = np.isin(general, self.ys_diffs)
        w = np.sum(ranges[indices])

        # report
        print('Критерий Вилкоксона')
        for alpha in self.alphas:
            u1 = self.m * (self.m + 1) / 2
            u3 = self.m * (self.n + self.m + 1) - self.omega
            u4 = self.m * self.n + self.m * (self.m + 1) / 2
            if (u1 <= w <= self.omega) or (u3 <= w <= u4):
                print(f'Гипотеза H0 отвергается на уровне значимости {alpha}, так как значение w = {w} попадает в критическую область [{u1}; {self.omega}] U [{u3}; {u4}]')
            else:
                print(f'Гипотеза H0 принимается на уровне значимости {alpha}, так как значение w = {w} не принадлежит критической области [{u1}; {self.omega}] U [{u3}; {u4}]')
            print()

    @staticmethod
    def edf(diffs: np.array):
        n = len(diffs)
        min_diffs = np.min(diffs)
        max_diffs = np.max(diffs)

        cum_freq = ss.cumfreq(diffs, numbins=n-1, defaultreallimits=(min_diffs, max_diffs))
        cum_freq = np.append([1], cum_freq[0])
        freq = cum_freq / n

        intervals = np.linspace(min_diffs, max_diffs, n)

        return sm.distributions.StepFunction(intervals, freq, side='right')

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


dict_data = LabData().lab_04_data

lab = Lab04(dict_data)
lab.processing()
