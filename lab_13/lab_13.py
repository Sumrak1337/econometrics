import numpy as np
import scipy.stats as ss

from data.lab_data import LabData


class Lab13:
    def __init__(self, main_data: dict):
        self.data = np.array([sample for sample in main_data.values()]).T
        self.rem = self.data[:, -1]

        self.alpha = 0.05

    def processing(self):
        ct1 = self.get_contigency_table(self.data[:, 0])
        ct2 = self.get_contigency_table(self.data[:, 1])

        _, p1 = ss.fisher_exact(ct1)
        _, p2 = ss.fisher_exact(ct2)

        # report
        print('Точный критерий Фишера')

        if p1 > self.alpha:
            print(f'Гипотеза H0 о взаимосвязи ДЛТ и ремиссии принимается на уровне значимости {self.alpha}, так как p > alpha ({p1:.4} > {self.alpha})')
        else:
            print(f'Гипотеза H0 о взаимосвязи ДЛТ и ремиссии отвергается на уровне значимости {self.alpha}, так как p <= alpha ({p1:.4} <= {self.alpha})')

        print()
        if p2 > self.alpha:
            print(f'Гипотеза H0 о взаимосвязи стадии N и ремиссии принимается на уровне значимости {self.alpha}, так как p > alpha ({p2:.4} > {self.alpha})')
        else:
            print(f'Гипотеза H0 о взаимосвязи стадии N и ремиссии отвергается на уровне значимости {self.alpha}, так как p <= alpha ({p2:.4} <= {self.alpha})')

    def get_contigency_table(self, col):
        matrix = np.zeros((2, 2))
        for i, j in zip(col, self.rem):
            matrix[i, j] += 1
        return matrix


dict_data = LabData().lab_13()
Lab13(dict_data).processing()
