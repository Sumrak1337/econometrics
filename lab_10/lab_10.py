import numpy as np
import scipy.stats as ss

from data.lab_data import LabData
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'
# TODO: rewrite from Sonya


class Lab10:
    def __init__(self, main_data: dict, stream):
        self.stream = stream
        self.data = np.array([value for value in main_data.values()])

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Исходные данные\n')
        for k, v in main_data.items():
            data.write(f'{k}: {v}\n')
        data.close()

        self.n, self.k = self.data.shape
        self.alpha = 0.05
        self.mean = np.mean(self.data)
        self.row_mean = np.mean(self.data, axis=1)
        self.column_mean = np.mean(self.data, axis=0)

        self.stream.write(f'Средние по столбцам: {self.column_mean.round(3)}\n')
        self.stream.write(f'Средние по строкам: {self.row_mean.round(3)}\n\n')

    def processing(self):
        s1 = self.n * np.sum((self.row_mean - self.mean) ** 2)
        s2 = self.k * np.sum((self.column_mean - self.mean) ** 2)
        s3 = 0
        for i in range(self.n):
            for j in range(self.k):
                s3 += (self.data[i][j] - self.row_mean[i] - self.column_mean[j] + self.mean) ** 2

        # report
        self.stream.write('Двухфакторный дисперсионный анализ\n')
        self.stream.write('1ая гипотеза: H0: b_i = 0\n')
        self.stream.write('2ая гипотеза: H0: t_i = 0\n\n')

        f1 = (s1 / (self.n - 1)) / (s3 / ((self.n - 1) * (self.k - 1)))
        f2 = (s2 / (self.k - 1)) / (s3 / ((self.n - 1) * (self.k - 1)))
        f_crit1 = ss.f.ppf(1 - self.alpha, dfn=self.n-1, dfd=(self.n-1)*(self.k-1))
        f_crit2 = ss.f.ppf(1 - self.alpha, dfn=self.k-1, dfd=(self.n-1)*(self.k-1))

        if f1 >= f_crit1:
            self.stream.write(f'Вторая нулевая гипотеза H0 отвергается \nна уровне значимости {self.alpha} при условии принятия первой нулевой гипотезы, \nтак как {f1:.3f} >= {f_crit1:.3f}\n\n')
        else:
            self.stream.write(f'Вторая нулевая гипотеза H0 принимается \nна уровне значимости {self.alpha} при условии принятия первой нулевой гипотезы, \nтак как {f1:.3f} < {f_crit1:.3f}\n\n')

        if f2 >= f_crit2:
            self.stream.write(f'Первая нулевая гипотеза H0 отвергается \nна уровне значимости {self.alpha} при условии принятия второй нулевой гипотезы, \nтак как {f2:.3f} >= {f_crit2:.3f}\n\n')
        else:
            self.stream.write(f'Первая нулевая гипотеза H0 принимается \nна уровне значимости {self.alpha} при условии принятия второй нулевой гипотезы, \nтак как {f2:.3f} < {f_crit2:.3f}\n\n')


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
dict_data = LabData.lab_10()
Lab10(dict_data, file).processing()
file.close()
