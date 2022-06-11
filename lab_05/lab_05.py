import numpy as np
import statsmodels.api as sm
import scipy.stats as ss

from data.lab_data import LabData


class Lab05:

    def __init__(self, main_data: dict):
        self.xs = np.array([value[0] for value in main_data.values()])
        self.ys = np.array([value[1] for value in main_data.values()])

        self.xs_diffs = np.sort(np.array([self.xs[i] - self.xs[i - 1] for i in range(1, len(self.xs))]).round(decimals=2))
        self.ys_diffs = np.sort(np.array([self.ys[i] - self.ys[i - 1] for i in range(1, len(self.ys))]).round(decimals=2))
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
        print('Экспорт')
        self.kolmogorov(self.xs_diffs)
        print('Импорт')
        self.kolmogorov(self.ys_diffs)

    def student_known_variances(self):
        phi_1 = (np.mean(self.xs_diffs) - np.mean(self.ys_diffs)) / np.sqrt(self.xs_disp / self.n + self.ys_disp / self.m)

        # report
        print('Критерий Стьюдента, известные дисперсии')

        if np.abs(phi_1) > self.quantile:
            print(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, так как значение phi1 = |{phi_1:.4}| принадлежит интервалу ({self.quantile}; +inf)')
        else:
            print(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, так как значение phi1 = |{phi_1:.4}| не принадлежит интервалу ({self.quantile}; +inf)')
        print()

    def student_unknown_variances(self):
        s = np.sqrt((self.n * self.xs_disp + self.m * self.ys_disp) / (self.n + self.m - 2))
        phi_2 = (self.xs_mean - self.ys_mean) / (s * np.sqrt(1 / self.n + 1 / self.m))

        # report
        print('Критерий Стьюдента, неизвестные дисперсии')

        if np.abs(phi_2) > self.quantile:
            print(
                f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, так как значение phi2 = |{phi_2:.4}| принадлежит интервалу ({self.quantile}; +inf)')
        else:
            print(
                f'Гипотеза H0 принимается на уровне значимости {self.alpha}, так как значение phi2 = |{phi_2:.4}| не принадлежит интервалу ({self.quantile}; +inf)')
        print()

    def fisher_snedekor(self):
        f = np.var(self.xs_diffs, ddof=1) / np.var(self.ys_diffs, ddof=1)
        u1 = 0.4994
        u2 = 2.0023

        # report
        print('Критерий Стьюдента, неизвестные дисперсии')

        if (0 <= f < u1) or (u2 < f < np.inf):
            print(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, так как значение F = {f:.4} принадлежит интервалу [0; {u1}) U ({u2}; +inf)')
        else:
            print(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, так как значение F = {f:.4} не принадлежит интервалу [0; {u1}) U ({u2}; +inf)')
        print()

    def kolmogorov(self, diffs):
        min_diffs = np.min(diffs)
        max_diffs = np.max(diffs)
        cum_freq = ss.cumfreq(diffs, numbins=self.n - 1, defaultreallimits=(min_diffs, max_diffs))
        cum_freq = np.append([1], cum_freq[0])
        freq = cum_freq / len(diffs)

        intervals = np.linspace(min_diffs, max_diffs, self.n)
        edf = sm.distributions.StepFunction(intervals, freq, side='right')

        left_value = np.max(np.array([i / self.n - edf.y[i] for i in range(1, self.n + 1)]))
        right_value = np.max(np.array([edf.y[i] - (i - 1) / self.n for i in range(1, self.n + 1)]))
        dn = max(left_value, right_value)
        dn = dn * (np.sqrt(self.n) - 0.01 + 0.85 / np.sqrt(self.n))

        # report
        print('Критерий Колмогорова')
        d_star = self.mod_quantiles[self.alpha][0]

        if dn <= d_star:
            print(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, так как значение {dn:.6} не попадает в интервал ({d_star}; +inf)')
        else:
            print(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, так как значение {dn:.6} попадает в интервал ({d_star}; +inf)')
        print()


dict_data = LabData().lab_05_data
Lab05(dict_data).processing()
