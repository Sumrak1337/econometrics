import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Lab12:
    def __init__(self, stream):
        self.stream = stream
        self.rs = np.random.RandomState(42)
        self.a_x = self.rs.randint(0, 100)
        self.a_y = self.rs.randint(0, 100)
        self.sigma_x = self.rs.randint(1, 150)
        self.sigma_y = self.rs.randint(1, 150)
        self.r = self.rs.randint(2, 20)
        self.s = self.rs.randint(2, 20)
        self.alpha = 0.05

        self.stream.write('Параметры для формирования выборок\n')
        self.stream.write(f'{"a_x":<{5}}: {self.a_x}\n')
        self.stream.write(f'{"a_y":<{5}}: {self.a_y}\n')
        self.stream.write(f'{"sigma_x":<{5}}: {self.sigma_x}\n')
        self.stream.write(f'{"sigma_y":<{5}}: {self.sigma_y}\n\n')
        self.stream.write(f'r: {self.r}\n')
        self.stream.write(f's: {self.s}\n\n')

        self.xs = ss.norm.ppf(self.rs.random(200), loc=self.a_x, scale=self.sigma_x)
        self.ys = ss.norm.ppf(self.rs.random(200), loc=self.a_y, scale=self.sigma_y)

        self.data = np.column_stack((self.xs, self.ys))

        data = open(RESULT_ROOT / 'data.txt', 'w', encoding='utf-8')
        data.write('Полученные выборки\n')
        data.write(f'{"X":<{6}} Y\n')
        for x, y in zip(self.xs, self.ys):
            data.write(f'{x:.2f}, {y:.2f}\n')
        data.close()

        self.lower = np.min(self.data, axis=0)
        self.upper = np.max(self.data, axis=0)
        self.blocks = np.array([self.s - 1, self.r - 1])

    def processing(self):
        grid = self.convert_to_grid()
        coord_cells = np.array([self.cell_to_z(cell) for cell in grid])
        ct = self.get_contigency_table(coord_cells)

        table = open(RESULT_ROOT / 'table.txt', 'w', encoding='utf-8')
        table.write('Таблица сопряжённости\n')
        table.write(f'{ct.astype(int)}')
        table.close()

        plt.figure()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(self.xs, self.ys, alpha=0.5)
        s = np.linspace(np.min(self.xs), np.max(self.xs), self.s)
        r = np.linspace(np.min(self.ys), np.max(self.ys), self.r)
        plt.vlines(s, np.min(self.ys), np.max(self.ys), alpha=0.1, linestyles='--')
        plt.hlines(r, np.min(self.xs), np.max(self.xs), alpha=0.1, linestyles='--')
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / 'scatters')
        plt.close('all')

        row_sum = np.sum(ct, axis=1)
        col_sum = np.sum(ct, axis=0)

        X2 = 0
        Y2 = 0
        for i in range(len(row_sum)):
            for j in range(len(col_sum)):
                exp_freq = row_sum[i] * col_sum[j] / np.sum(ct)
                X2 += (ct[i][j] - exp_freq) ** 2 / exp_freq
                if ct[i][j] != 0:
                    Y2 += ct[i][j] * np.log(ct[i][j] / exp_freq)
        Y2 = 2 * Y2
        chi2 = ss.chi2.ppf(1 - self.alpha, df=(self.r-1)*(self.s-1))

        # report
        self.stream.write('Гипотеза о независимости номинальных признаков\n')

        if X2 < chi2:
            self.stream.write(f'Гипотеза H0 о независимости признаков принимается на уровне значимости {self.alpha}, \nтак как значение X2 < chi2 ({X2:.4} < {chi2:.4})\n\n')
        else:
            self.stream.write(f'Гипотеза H0 о независимости признаков отклоняется на уровне значимости {self.alpha}, \nтак как значение X2 >= chi2 ({X2:.4} >= {chi2:.4})\n\n')

        if Y2 < chi2:
            self.stream.write(f'Гипотеза H0 о независимости признаков принимается на уровне значимости {self.alpha}, \nтак как значение Y2 < chi2 ({Y2:.4} < {chi2:.4})\n\n')
        else:
            self.stream.write(f'Гипотеза H0 о независимости признаков отклоняется на уровне значимости {self.alpha}, \nтак как значение Y2 >= chi2 ({Y2:.4} >= {chi2:.4})\n\n')

    @staticmethod
    def get_contigency_table(coord_cells):
        r = np.max(coord_cells, axis=0)
        matrix = np.zeros((r[0] + 1, r[1] + 1))
        for point in coord_cells:
            x = point[0]
            y = point[1]
            matrix[x, y] += 1
        return matrix

    def convert_to_grid(self):
        block_widths = (self.upper - self.lower) / self.blocks
        _, dim = self.data.shape
        cp = np.ones(len(self.blocks) + 1)
        np.cumprod(self.blocks, out=cp[1:])

        res = []
        for row in self.data:
            z = row
            cell_id = np.array(cp[:dim] * np.floor((z - self.lower) / block_widths))
            res.append(np.sum(cell_id - cp[:dim] * (z == self.upper)))
        return np.array(res).astype(int)

    def cell_to_z(self, cell):
        coord = np.array([])

        for i in range(len(self.blocks)):
            coord = np.append(coord, cell % self.blocks[i])
            cell = cell // self.blocks[i]
        return coord.astype(int)


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
Lab12(file).processing()
file.close()
