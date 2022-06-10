import numpy as np
import statsmodels.api as sm
import scipy.stats as ss

from data.lab_data import LabData


class Lab03:

    def __init__(self, main_data: dict):
        self.data = np.array([])
        for value in main_data.values():
            self.data = np.append(self.data, value)
        self.alphas = [0.05, 0.1]
        self.mod_quantiles = {0.15: [0.775, 0.091],
                              0.10: [0.819, 0.104],
                              0.05: [0.895, 0.126],
                              0.025: [0.955, 0.148],
                              0.01: [1.035, 0.178]}
        self.n = None
        self.mean = None
        self.var = None

    def processing(self):
        diffs = np.array([self.data[i] - self.data[i - 1] for i in range(1, len(self.data))])
        diffs = np.unique(diffs.round(decimals=2))
        self.n = len(diffs)

        self.mean = np.mean(diffs)
        self.var = np.var(diffs, ddof=1)

        min_diffs = np.min(diffs)
        max_diffs = np.max(diffs)
        cum_freq = ss.cumfreq(diffs, numbins=self.n-1, defaultreallimits=(min_diffs, max_diffs))
        cum_freq = np.append([1], cum_freq[0])
        freq = cum_freq / self.n

        intervals = np.linspace(min_diffs, max_diffs, self.n)
        ecdf = sm.distributions.StepFunction(intervals, freq, side='right')

        self.kolmogorov(ecdf)
        self.omega(ecdf)

    def kolmogorov(self, edf):
        left_value = np.max(np.array([i / self.n - edf.y[i] for i in range(1, self.n + 1)]))
        right_value = np.max(np.array([edf.y[i] - (i - 1) / self.n for i in range(1, self.n + 1)]))
        dn = max(left_value, right_value)
        dn = dn * (np.sqrt(self.n) - 0.01 + 0.85 / np.sqrt(self.n))

        # report
        print('Критерий Колмогорова')

        for alpha in self.alphas:
            answer = False
            d_star = self.mod_quantiles[alpha][0]

            if dn <= d_star:
                answer = True

            if answer:
                print(f'Гипотеза H0 принимается на уровне значимости {alpha}, так как значение {dn:.8} не попадает в интервал ({d_star}; +inf)')
            else:
                print(f'Гипотеза H0 отвергается на уровне значимости {alpha}, так как значение {dn:.8} попадает в интервал ({d_star}; +inf)')
        print()

    def omega(self, edf):
        s = 0
        for i in range(1, self.n + 1):
            s += (edf.y[i] - (2 * i - 1) / (2 * self.n)) ** 2
        omega_sqr = 1 / (12 * self.n) + s

        omega_sqr = omega_sqr * (1 + 0.5 / self.n)

        # report
        print('Критерий omega^2')

        for alpha in self.alphas:
            answer = False
            omega_star = self.mod_quantiles[alpha][1]

            if omega_sqr <= omega_star:
                answer = True

            if answer:
                print(f'Гипотеза H0 принимается на уровне значимости {alpha}, так как значение {omega_sqr:.8} не попадает в интервал ({omega_star}; +inf)')
            else:
                print(f'Гипотеза H0 отвергается на уровне значимости {alpha}, так как значение {omega_sqr:.8} попадает в интервал ({omega_star}; +inf)')


if __name__ == '__main__':
    dict_data = LabData().lab_03()

    Lab03(dict_data).processing()
