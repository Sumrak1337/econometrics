import numpy as np

from data.lab_data import LabData


class Lab06:
    def __init__(self, main_data: dict):
        self.xs = np.array([value for value in main_data.values()])
        self.xs_diffs = np.array([self.xs[i] - self.xs[i - 1] for i in range(1, len(self.xs))])

        self.teta = 1.5
        self.alpha = 0.05
        self.quantile = 1.96

    def processing(self):
        diffs = self.xs_diffs - self.teta
        diffs = diffs[diffs != 0]
        s = np.sum(diffs > 0)
        n = len(diffs)
        s_star = (s - n / 2) / np.sqrt(n / 4)

        # report
        print('Критерий знаков для одной выборки')

        if (-np.inf < s_star < -self.quantile) or (self.quantile < s_star < np.inf):
            print(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, так как значение S* = {s_star:.4} '
                  f'принадлежит критической области (-inf; {-self.quantile}) U ({self.quantile}; +inf).')
        else:
            print(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, так как значение S* = {s_star:.4} '
                  f'не принадлежит критической области (-inf; {-self.quantile}) U ({self.quantile}; +inf).')


dict_data = LabData.lab_06()
Lab06(dict_data).processing()
