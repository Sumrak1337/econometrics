import numpy as np

from data.lab_data import LabData
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'
# TODO: add quantiles


class Lab06:
    def __init__(self, main_data: dict, stream):
        self.stream = stream
        self.xs = np.array([value for value in main_data.values()])

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Изначальные данные\n')
        for k, v in main_data.items():
            data.write(f'{k}: {v}\n')
        data.close()

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
        self.stream.write('Критерий знаков для одной выборки:\n')

        if (-np.inf < s_star < -self.quantile) or (self.quantile < s_star < np.inf):
            self.stream.write(f'Гипотеза H0 отвергается на уровне значимости {self.alpha}, \nтак как значение S* = {s_star:.4} '
                              f'принадлежит критической области \n(-inf; {-self.quantile}) U ({self.quantile}; +inf).')
        else:
            self.stream.write(f'Гипотеза H0 принимается на уровне значимости {self.alpha}, \nтак как значение S* = {s_star:.4} '
                              f'не принадлежит критической области \n(-inf; {-self.quantile}) U ({self.quantile}; +inf).')


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
dict_data = LabData.lab_06()
Lab06(dict_data, file).processing()
file.close()
