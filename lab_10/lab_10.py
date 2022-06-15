import numpy as np
import scipy.stats as ss

from data.lab_data import LabData


class Lab10:
    def __init__(self, main_data: dict):
        self.data = np.array([value for value in main_data.values()])

        self.n, self.k = self.data.shape
        self.alpha = 0.05
        self.mean = np.mean(self.data)
        self.row_mean = np.mean(self.data, axis=1)
        self.column_mean = np.mean(self.data, axis=0)

    def processing(self):
        s1 = self.n * np.sum((self.row_mean - self.mean) ** 2)
        s2 = self.k * np.sum((self.column_mean - self.mean) ** 2)
        s3 = 0
        for i in range(self.n):
            for j in range(self.k):
                s3 += (self.data[i][j] - self.row_mean[i] - self.column_mean[j] + self.mean) ** 2

        # report
        print('Двухфакторный дисперсионный анализ')
        print('1ая гипотеза: H0: b_i = 0')
        print('2ая гипотеза: H0: t_i = 0')

        f1 = (s1 / (self.n - 1)) / (s3 / ((self.n - 1) * (self.k - 1)))
        f2 = (s2 / (self.k - 1)) / (s3 / ((self.n - 1) * (self.k - 1)))
        f_crit1 = ss.f.ppf(1 - self.alpha, dfn=self.n-1, dfd=(self.n-1)*(self.k-1))
        f_crit2 = ss.f.ppf(1 - self.alpha, dfn=self.k-1, dfd=(self.n-1)*(self.k-1))

        if f1 >= f_crit1:
            print(f'Вторая нулевая гипотеза H0 отвергается на уровне значимости {self.alpha} при условии принятия первой нулевой гипотезы, так как {f1} >= {f_crit1}')
        else:
            print(f'Вторая нулевая гипотеза H0 принимается на уровне значимости {self.alpha} при условии принятия первой нулевой гипотезы, так как {f1} < {f_crit1}')

        if f2 >= f_crit2:
            print(f'Первая нулевая гипотеза H0 отвергается на уровне значимости {self.alpha} при условии принятия второй нулевой гипотезы, так как {f2} >= {f_crit2}')
        else:
            print(f'Первая нулевая гипотеза H0 принимается на уровне значимости {self.alpha} при условии принятия второй нулевой гипотезы, так как {f2} < {f_crit2}')


dict_data = LabData.lab_10()
Lab10(dict_data).processing()
