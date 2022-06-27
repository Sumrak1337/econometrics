import numpy as np
import scipy.stats as ss

from data.lab_data import LabData
from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Lab13:
    def __init__(self, main_data: dict, stream):
        self.stream = stream
        self.data = np.array([sample for sample in main_data.values()]).T
        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write(f'{list(main_data)}\n')
        data.write(f'{self.data}\n')
        data.close()

        self.rem = self.data[:, -1]

        self.alpha = 0.05

    def processing(self):
        ct1 = self.get_contigency_table(self.data[:, 0])
        ct2 = self.get_contigency_table(self.data[:, 1])

        table = open(RESULT_ROOT / 'table.txt', 'w', encoding='utf-8')
        table.write('ДЛТ и ремиссия\n')
        table.write(f'{ct1.astype(int)}\n\n')
        table.write('N и ремиссия\n')
        table.write(f'{ct2.astype(int)}\n')
        table.close()

        _, p1 = ss.fisher_exact(ct1)
        _, p2 = ss.fisher_exact(ct2)

        self.stream.write('P-значения точного критерия Фишера\n')
        self.stream.write(f'p1: {p1:.4}\n')
        self.stream.write(f'p2: {p2:.4}\n\n')

        # report
        self.stream.write('Точный критерий Фишера\n')

        if p1 > self.alpha:
            self.stream.write(f'Гипотеза H0 о взаимосвязи ДЛТ и ремиссии принимается на уровне значимости {self.alpha}, \nтак как p > alpha ({p1:.4} > {self.alpha})\n\n')
        else:
            self.stream.write(f'Гипотеза H0 о взаимосвязи ДЛТ и ремиссии отвергается на уровне значимости {self.alpha}, \nтак как p <= alpha ({p1:.4} <= {self.alpha})\n\n')

        if p2 > self.alpha:
            self.stream.write(f'Гипотеза H0 о взаимосвязи стадии N и ремиссии принимается на уровне значимости {self.alpha}, \nтак как p > alpha ({p2:.4} > {self.alpha})\n\n')
        else:
            self.stream.write(f'Гипотеза H0 о взаимосвязи стадии N и ремиссии отвергается на уровне значимости {self.alpha}, \nтак как p <= alpha ({p2:.4} <= {self.alpha})\n\n')

    def get_contigency_table(self, col):
        matrix = np.zeros((2, 2))
        for i, j in zip(col, self.rem):
            matrix[i, j] += 1
        return matrix


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
dict_data = LabData().lab_13()
Lab13(dict_data, file).processing()
file.close()
