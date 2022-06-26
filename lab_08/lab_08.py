import numpy as np
import scipy.stats as ss

from data.lab_data import LabData
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Lab08:
    def __init__(self, main_data: dict, stream):
        self.stream = stream
        self.samples = [value for value in main_data.values()]

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Исходные данные\n')
        for k, v in main_data.items():
            data.write(f'{k}: {v}\n')
        data.close()

        self.Q1 = None
        self.Q2 = None
        self.Q = None
        self.k = len(self.samples)
        self.n = np.sum([len(value) for value in self.samples])
        self.alpha = 0.05
        self.c = np.linspace(-1, 1, self.k)
        self.means = np.array([np.mean(value) for value in self.samples])

        self.global_mean = self.get_global_mean()

    def processing(self):
        q1 = 0
        q2 = 0
        q = 0
        for i in range(self.k):
            q1 += self.q1(self.samples[i])
            q2 += self.q2(self.samples[i])
            q += self.q(self.samples[i])

        self.Q1 = q1
        self.Q2 = q2
        self.Q = q

        self.stream.write(f'Q1 = {self.Q1:.3f}\n')
        self.stream.write(f'Q2 = {self.Q2:.3f}\n')
        self.stream.write(f'Q = Q1 + Q2 = {self.Q:.3f}\n\n')
        critical_value = ss.f.ppf(1 - self.alpha, dfn=self.k-1, dfd=self.n-self.k)
        intergroup_var = q1 / (self.k - 1)
        intragoup_var = q2 / (self.n - self.k)
        f_value = intergroup_var / intragoup_var

        # report
        self.stream.write('Критерий Фишера\n')
        if f_value > critical_value:
            self.stream.write(f'Гипотеза H0 о равенстве средних отклоняется, \nтак как F_наблюдаемое > F_критическое ({f_value:.4} > {critical_value:.4})\n\n')
            self.linear_contrast_method([self.samples[0], self.samples[1]])
            self.linear_contrast_method([self.samples[0], self.samples[2]])
            self.linear_contrast_method([self.samples[1], self.samples[2]])
            self.linear_contrast_method([self.samples[0], self.samples[1], self.samples[2]])
        else:
            self.stream.write(f'Гипотеза H0 о равенстве средних принимается, \nтак как F_наблюдаемое <= F_критическое ({f_value:.4} <= {critical_value:.4}\n\n)')

    def get_global_mean(self):
        len_sum = 0
        val_sum = 0
        for i in range(self.k):
            val_sum += np.sum(self.samples[i])
            len_sum += np.sum(len(self.samples[i]))
        return val_sum / len_sum

    def q1(self, data: np.array):
        return len(data) * (np.mean(data) - self.global_mean) ** 2

    @staticmethod
    def q2(data: np.array):
        return np.sum((data - np.mean(data)) ** 2)

    def q(self, data: np.array):
        return np.sum((data - self.global_mean) ** 2)

    def linear_contrast_method(self, s: list):
        k = len(s)
        Q2 = 0
        for sample in s:
            Q2 += self.q2(sample)
        n = np.sum([len(sample) for sample in s])
        c = np.linspace(-1, 1, k)
        est_lin_contrast = np.sum([coef * np.mean(sample) for coef, sample in zip(c, s)])
        est_disp = Q2 * np.sum([coef ** 2 / len(sample) for coef, sample in zip(c, s)]) / (n - k)
        left, right = self.get_conf_ints(est_lin_contrast, est_disp, k, n)

        # report
        if k == 2:
            if left < np.mean(s[0]) - np.mean(s[1]) < right:
                self.stream.write(f'Гипотеза H0 о равенстве двух средних принимается, \nтак как ноль содержится в доверительном интервале ({left:.4}; {right:.4})\n\n')
            else:
                self.stream.write(f'Гипотеза H0 о равенстве двух средних отвергается, \nтак как ноль не содержится в доверительном интервале ({left:.4}; {right:.4})\n\n')

        else:
            m1 = np.mean(s[0])
            m2 = np.mean(s[1])
            m3 = np.mean(s[2])
            if left < 0.5 * (m1 + m3) - m2 < right:
                self.stream.write(f'Гипотеза H0 о равенстве средних принимается, \nтак как ноль содержится в доверительном интервале ({left:.4}; {right:.4})\n\n')
            else:
                self.stream.write(f'Гипотеза H0 о равенстве средних отвергается, \nтак как ноль не содержится в доверительном интервале ({left:.4}; {right:.4})\n\n')

    def get_conf_ints(self, elc, ed, k, n):
        f = ss.f.ppf(self.alpha, dfn=k-1, dfd=n-k)
        left = elc - np.sqrt(ed * (k - 1) * f)
        right = elc + np.sqrt(ed * (k - 1) * f)
        return left, right


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
dict_data = LabData().lab_08_data
Lab08(dict_data, file).processing()
file.close()
