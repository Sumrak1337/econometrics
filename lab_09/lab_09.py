import numpy as np
import scipy.stats as ss

from data.lab_data import LabData


class Lab09:
    def __init__(self, main_data: np.array):
        self.data = [sorted(value) for value in main_data.values()]
        self.gen_data = sorted([item for lst in self.data for item in lst])

        self.k = len(self.data)
        self.n = len(self.gen_data)
        self.alpha = 0.05

        self.ranges = self.get_ranges(self.gen_data)
        self.general_mean_rank = (self.n + 1) / 2
        self.mean_ranges = np.array([])

    def processing(self):
        for i in range(self.k):
            indices = np.isin(self.gen_data, self.data[i])
            self.mean_ranges = np.append(self.mean_ranges, np.mean(self.ranges[indices]))

        H = 12 * self.n * (np.mean(self.mean_ranges) - self.general_mean_rank) ** 2 / self.n / (self.n + 1)
        H_crit = ss.chi2.ppf(q=1-self.alpha, df=self.k-1)

        # report
        if H > H_crit:
            print(f'Гипотеза H0 об однородности выборок на уровне значимости {self.alpha} отклоняется, '
                  f'так как статистика H больше критического значения распределения хи-квадрата ({H:.4} > {H_crit:.4})')
        else:
            print(f'Гипотеза H0 об однородности выборок на уровне значимости {self.alpha} принимается, '
                  f'так как статистика H меньше критического значения распределения хи-квадрата ({H:.4} < {H_crit:.4})')

    @staticmethod
    def get_ranges(general):
        unique, indices, counts = np.unique(general, return_index=True, return_counts=True)
        res = np.array([])
        for cnt, idx in zip(counts, indices):
            if cnt > 1:
                value = (2 * idx + cnt + 1) / 2
            else:
                value = idx + 1
            res = np.append(res, [value] * cnt)

        return res


dict_data = LabData.lab_08()
Lab09(dict_data).processing()
