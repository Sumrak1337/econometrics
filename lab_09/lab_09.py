import numpy as np
import scipy.stats as ss

from data.lab_data import LabData
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Lab09:
    def __init__(self, main_data: dict, stream):
        self.stream = stream

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Исходные данные\n')
        for k, v in main_data.items():
            data.write(f'{k}: {v}\n')
        data.close()

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

        s = 0
        for i in range(self.k):
            s += len(self.data[i]) * (self.mean_ranges[i] - self.general_mean_rank) ** 2
        H = 12 * s / self.n / (self.n + 1)
        H_crit = ss.chi2.ppf(q=1-self.alpha, df=self.k-1)

        # report
        if H > H_crit:
            self.stream.write(f'Гипотеза H0 об однородности выборок на уровне значимости {self.alpha} отклоняется, '
                              f'\nтак как статистика H больше критического значения распределения хи-квадрата \n({H:.4f} > {H_crit:.4})\n')
        else:
            self.stream.write(f'Гипотеза H0 об однородности выборок на уровне значимости {self.alpha} принимается, '
                              f'\nтак как статистика H меньше критического значения распределения хи-квадрата \n({H:.4f} < {H_crit:.4})\n')

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


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
dict_data = LabData.lab_08()
Lab09(dict_data, file).processing()
file.close()
