import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss

from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'


class Lab01:
    def __init__(self, stream):
        self.stream = stream
        self.rs = np.random.RandomState(42)
        self.colors = list(mcolors.TABLEAU_COLORS)

        self.a1 = self.rs.randint(0, 10)
        self.a2 = self.rs.randint(0, 10)
        self.sigma1 = self.rs.randint(1, 15)
        self.sigma2 = self.rs.randint(1, 15)

        self.sample1 = ss.norm.ppf(self.rs.random(80), loc=self.a1, scale=self.sigma1).astype(int)
        self.sample2 = ss.norm.ppf(self.rs.random(60), loc=self.a2, scale=self.sigma2).astype(int)
        self.general_sample = np.append(self.sample1, self.sample2)
        self.general_sample = np.sort(self.general_sample)
        self.bins = int(np.ceil(1 + 3.2 * np.log10(len(self.general_sample))))  # 10

        self.n = None
        self.mid_intervals = None
        self.rel_freq = None
        self.rel_dens = None
        self.cum_freq = None
        self.rel_cum_freq = None

    def processing(self):
        self.describe(self.general_sample)
        self.n, self.mid_intervals = self.freq_hist()

        self.rel_freq_hist(self.n)
        self.rel_dens = self.rel_freq / np.diff(self.mid_intervals)[0]
        self.rel_den_hist()

        self.cum_freq = np.cumsum(self.n)
        self.rel_cum_freq = self.cum_freq / len(self.general_sample)
        self.rel_cum_hist()

        self.freq_hist_cum_freq(self.bins)

        self.correlation()

    def freq_hist(self):
        plt.figure()
        plt.title('Гистограмма частот распределения')
        n, intervals, _ = plt.hist(self.general_sample, rwidth=0.9, bins=self.bins)
        mid_intervals = np.array([(intervals[i] + intervals[i + 1]) / 2 for i in range(self.bins)])
        plt.plot(mid_intervals, n)
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('frequency')
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'freq_hist_{self.bins}_bins')
        plt.close('all')

        return n, mid_intervals

    def rel_freq_hist(self, n):
        plt.figure()
        plt.title('Гистограмма относительных частот')
        self.rel_freq = n / len(self.general_sample)
        plt.bar(self.mid_intervals, height=self.rel_freq, width=50/self.bins)
        plt.plot(self.mid_intervals, self.rel_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('relative frequency')
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'rel_freq_hist_{self.bins}_bins')
        plt.close('all')

    def rel_den_hist(self):
        plt.figure()
        plt.title('Гистограмма плотности относительных частот')
        plt.bar(self.mid_intervals, height=self.rel_dens, width=50/self.bins)
        plt.plot(self.mid_intervals, self.rel_dens, color=self.colors[1])
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('frequency density')
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'rel_den_hist_{self.bins}_bins')
        plt.close('all')

    def rel_cum_hist(self):
        plt.figure()
        plt.title('Гистограмма относительных кумулятивных частот')
        plt.bar(self.mid_intervals, height=self.rel_cum_freq, width=50 / self.bins)
        plt.plot(self.mid_intervals, self.rel_cum_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({self.bins} intervals)')
        plt.ylabel('relative frequency')
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'rel_cum_hist_{self.bins}_bins')
        plt.close('all')

    def freq_hist_cum_freq(self, bins=7):
        n, intervals, _ = plt.hist(self.general_sample, rwidth=0.9, bins=bins)
        plt.close('all')
        plt.figure()
        plt.title('Гистограмма распределения частот и график интегральных частот')
        mid_ints = np.array([(intervals[i] + intervals[i + 1]) / 2 for i in range(bins)])
        cum_freq = np.cumsum(n) / len(self.general_sample)
        plt.bar(mid_ints, height=n / len(self.general_sample), width=50 / bins)
        plt.plot(mid_ints, cum_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({bins} intervals)')
        plt.ylabel('frequency')
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'freq_hist_cum_freq_{bins}_bins')
        plt.close('all')

    def describe(self, sample: np.array):
        param_gap = 8
        left_gap = 35
        right_gap = 6
        self.stream.write('Выборочные характеристики\n\n')
        self.stream.write('Параметры для формирования выборок:\n')
        self.stream.write(f'{r"a_1":<{param_gap}}: {self.a1:<{param_gap}}\n')
        self.stream.write(f'{r"a_2":<{param_gap}}: {self.a2:<{param_gap}}\n')
        self.stream.write(f'{r"sigma_1":<{param_gap}}: {self.sigma1:<{param_gap}}\n')
        self.stream.write(f'{r"sigma_2":<{param_gap}}: {self.sigma2:<{param_gap}}\n')
        self.stream.write('\n')

        self.stream.write(f'Выборка:\n {sample.reshape(20, 7)}\n')
        self.stream.write(f'{"Максимум":<{left_gap}}: {np.max(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Минимум":<{left_gap}}: {np.min(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Число наблюдений":<{left_gap}}: {len(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Среднее значение":<{left_gap}}: {np.mean(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Медиана":<{left_gap}}: {np.median(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Мода":<{left_gap}}: {ss.mode(sample)[0][0]:>{right_gap}.2f}\n')
        self.stream.write(f'{"Размах":<{left_gap}}: {np.max(sample) - np.min(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Несмещённая выборочная дисперсия":<{left_gap}}: {np.var(sample, ddof=1):>{right_gap}.2f}\n')
        self.stream.write(f'{"Смещённая выборочная дисперсия":<{left_gap}}: {np.var(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Несмещённое стандартное отклонение":<{left_gap}}: {np.std(sample, ddof=1):>{right_gap}.2f}\n')
        self.stream.write(f'{"Смещённое стандартное отклонение":<{left_gap}}: {np.std(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Среднее абсолютное отклонение":<{left_gap}}: {np.mean(np.abs(sample - np.mean(sample))):>{right_gap}.2f}\n')
        self.stream.write(f'{"Эксцесс":<{left_gap}}: {ss.kurtosis(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Ассиметрия":<{left_gap}}: {ss.skew(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Коэффициент вариации":<{left_gap}}: {np.std(sample, ddof=1) / np.mean(sample):>{right_gap}.2f}\n')
        self.stream.write(f'{"Ошибка выборки":<{left_gap}}: {np.std(sample, ddof=1) / np.sqrt(len(sample)):>{right_gap}.2f}\n')
        self.stream.write('\n')

    def correlation(self):
        sample1 = self.rs.uniform(0, 1, 140)
        a = self.rs.randint(1, 5)
        sigma = self.rs.randint(7, 15)
        sample2 = ss.norm.ppf(sample1, loc=a, scale=sigma)
        sample3 = np.array([self.rs.normal(10, 2) for _ in range(140)])

        r1 = self.corr(sample1, sample2)
        r2 = self.corr(sample2, sample3)
        r3 = self.corr(sample3, sample1)

        stat_signif1 = self.statistical_significance(r1, len(sample1))
        stat_signif2 = self.statistical_significance(r2, len(sample2))
        stat_signif3 = self.statistical_significance(r3, len(sample3))

        cov1 = np.cov(sample1, sample2)[0, 1]
        cov2 = np.cov(sample2, sample3)[0, 1]
        cov3 = np.cov(sample3, sample1)[0, 1]

        std1 = np.std(sample1, ddof=1)
        std2 = np.std(sample2, ddof=1)
        std3 = np.std(sample3, ddof=1)

        alpha = 0.05
        # TODO: change critical value
        critical_value = 1.96

        corr_matrix = np.corrcoef(np.array([sample1, sample2, sample3]))

        # report
        self.stream.write('Корреляция\n\n')
        self.stream.write('Параметры для формирования выборок:\n')
        self.stream.write(f'{"a":<{5}}: {a}\n')
        self.stream.write(f'{"sigma":<{5}}: {sigma}\n\n')

        self.stream.write('Выборочные коэффициенты корреляции:\n')
        self.stream.write(f'r1 = {r1:.3f}\n')
        self.stream.write(f'r2 = {r2:.3f}\n')
        self.stream.write(f'r3 = {r3:.3f}\n')
        self.stream.write('\n')
        self.stream.write('Статистическая значимость вычисленных парных коэффициентов корреляции:\n')
        if np.abs(stat_signif1) > critical_value:
            self.stream.write(f'1. Гипотеза о равенстве нулю корреляции отклоняется на уровне значимости {alpha},\n так как |{stat_signif1:.3f}| > {critical_value:.3f}\n')
        else:
            self.stream.write(f'1. Гипотеза о равенстве нулю корреляции принимается на уровне значимости {alpha},\n так как |{stat_signif1:.3f}| <= {critical_value:.3f}\n')

        if np.abs(stat_signif2) > critical_value:
            self.stream.write(f'2. Гипотеза о равенстве нулю корреляции отклоняется на уровне значимости {alpha},\n так как |{stat_signif2:.3f}| > {critical_value:.3f}\n')
        else:
            self.stream.write(f'2. Гипотеза о равенстве нулю корреляции принимается на уровне значимости {alpha},\n так как |{stat_signif2:.3f}| <= {critical_value:.3f}\n')

        if np.abs(stat_signif3) > critical_value:
            self.stream.write(f'3. Гипотеза о равенстве нулю корреляции отклоняется на уровне значимости {alpha},\n так как |{stat_signif3:.3f}| > {critical_value:.3f}\n')
        else:
            self.stream.write(f'3. Гипотеза о равенстве нулю корреляции принимается на уровне значимости {alpha},\n так как |{stat_signif3:.3f}| <= {critical_value:.3f}\n')
        self.stream.write('\n')

        self.stream.write('Выборочные ковариации для пар выборок:\n')
        self.stream.write(f'cov(1, 2) = {cov1:.3f}\n')
        self.stream.write(f'cov(2, 3) = {cov2:.3f}\n')
        self.stream.write(f'cov(3, 1) = {cov3:.3f}\n')
        self.stream.write('\n')

        self.stream.write('Выборочные средние квадратические отклонения:\n')
        self.stream.write(f'sd1 = {std1:.3f}\n')
        self.stream.write(f'sd2 = {std2:.3f}\n')
        self.stream.write(f'sd3 = {std3:.3f}\n')
        self.stream.write('\n')

        self.stream.write('Корреляционная матрица 3х3:\n')
        np.set_printoptions(precision=2)
        self.stream.write(f'{corr_matrix}\n')

    @staticmethod
    def corr(s1: np.array, s2: np.array):
        return np.corrcoef(s1, s2)[0, 1]

    @staticmethod
    def statistical_significance(r, n):
        return r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
Lab01(file).processing()
file.close()
