import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as ss

from pathlib import Path

RESULT_ROOT = Path(__file__).parent.resolve() / 'report' / 'figures'
# TODO: add critical values


class Lab02:
    def __init__(self, stream):
        self.stream = stream
        self.rs = np.random.RandomState(42)
        self.a = self.rs.randint(0, 10)
        self.sigma = self.rs.randint(1, 15)
        self.bins1 = 8
        self.bins2 = 12
        self.bins3 = 5
        self.bins4 = 7
        self.colors = list(mcolors.TABLEAU_COLORS)

        self.data = ss.norm.ppf(self.rs.random(200), loc=self.a, scale=self.sigma)

        self.disp = np.var(self.data, ddof=1)
        self.sd = np.std(self.data, ddof=1)
        self.mean = np.mean(self.data)

        self.sample1 = self.rs.choice(self.data, 30)
        self.mean1 = np.mean(self.sample1)
        self.disp1 = np.var(self.sample1, ddof=1)
        self.sd1 = np.std(self.sample1, ddof=1)

        self.sample2 = self.rs.choice(self.data, 30)
        self.mean2 = np.mean(self.sample2)
        self.disp2 = np.var(self.sample2, ddof=1)
        self.sd2 = np.std(self.sample2, ddof=1)

    def processing(self):
        self.stream.write('Параметры для формирования выборки:\n')
        self.stream.write(f'{"a":<{5}}: {self.a}\n')
        self.stream.write(f'{"sigma":<{5}}: {self.sigma}\n\n')

        self.stream.write('Вычисленные значения генеральной совокупности:\n')
        self.stream.write(f'{"Дисперсия":<{23}}: {self.disp:.3}\n')
        self.stream.write(f'{"Математическое ожидание":<{23}}: {self.mean:.3}\n')
        self.stream.write(f'{"Стандартное отклонение":<{23}}: {self.sd:.3}\n\n')

        self.rel_freq_hist(self.data, self.bins1, 'data')
        self.rel_freq_hist(self.data, self.bins2, 'data')
        self.probability()

        self.stream.write(f'Точечные оценки параметров распределения для полученных 2х выборок:\n')
        self.stream.write(f'{" ":<{29}} sample1 | sample2\n')
        self.stream.write(f'{"Среднее значение":<{31}}: {self.mean1:.3} | {self.mean2:.3}\n')
        self.stream.write(f'{"Дисперсия":<{31}}: {self.disp1:.3} | {self.disp2:.3}\n')
        self.stream.write(f'{"Среднеквадратическое отклонение":<{31}}: {self.sd1:.3} | {self.sd2:.3}\n\n')

        self.rel_freq_hist(self.sample1, self.bins3, 'sample1')
        self.rel_freq_hist(self.sample1, self.bins4, 'sample1')
        self.rel_freq_hist(self.sample2, self.bins3, 'sample2')
        self.rel_freq_hist(self.sample2, self.bins4, 'sample2')

        for sample, name in zip([self.data, self.sample1, self.sample2], ['data', 'sample1', 'sample2']):
            self.mean_conf_interval_known_sd(sample, name)
            self.mean_conf_interval_unknown_sd(sample, name)
            self.disp_conf_interval_unknown_mean(sample, name)
            self.general_sd_conf_interval(sample, name)

        self.bernoulli()

    def rel_freq_hist(self, data: np.array, bins, name):
        n, intervals, _ = plt.hist(data, bins=bins)
        mid_intervals = np.array([(intervals[i] + intervals[i + 1]) / 2 for i in range(bins)])
        plt.close('all')

        plt.figure()
        plt.title(f'Гистограмма относительных частот ({name})')
        rel_freq = n / len(data)
        plt.bar(mid_intervals, height=rel_freq)
        plt.plot(mid_intervals, rel_freq, color=self.colors[1])
        plt.xlabel(f'x_value ({bins} intervals)')
        plt.ylabel('relative frequency')
        plt.tight_layout()
        plt.savefig(RESULT_ROOT / f'rel_freq_hist_{bins}_bins_{name}')
        plt.close('all')

    def probability(self):
        prob1 = ss.norm.pdf(11, loc=self.mean, scale=self.sd) - ss.norm.pdf(5, loc=self.mean, scale=self.sd)
        prob2 = ss.norm.pdf(11, loc=self.a, scale=self.sigma) - ss.norm.pdf(5, loc=self.a, scale=self.sigma)
        self.stream.write(f'Значение вероятности F(11) - F(5) при \nвычисленных a = {self.mean:.4} и sigma = {self.sd:.4} равно: {prob1:.5}\n')
        self.stream.write(f'Значение вероятности F(11) - F(5) при \nданных a = {self.a} и sigma = {self.sigma} равно: {prob2:.5}\n')
        self.stream.write(f'Абсолютная разность: {np.abs(prob1 - prob2):.4}\n\n')

    def mean_conf_interval_known_sd(self, data, name):
        left_edge = np.mean(data) - ss.norm(loc=0, scale=1).ppf(0.975) * np.std(data, ddof=1) / np.sqrt(len(data))
        right_edge = np.mean(data) + ss.norm(loc=0, scale=1).ppf(0.975) * np.std(data) / np.sqrt(len(data))
        self.stream.write(f'{name}\n')
        self.stream.write(f'Доверительный интервал для оценки \nмат. ожидания при известном СКО (уровень доверия = 0.95): \n({left_edge:.3}, {right_edge:.3})\n\n')

    def mean_conf_interval_unknown_sd(self, data, name):
        n = len(data)
        left_edge = np.mean(data) - ss.t.ppf(q=0.975, df=n - 1) * np.std(data, ddof=1) / np.sqrt(n)
        right_edge = np.mean(data) + ss.t.ppf(q=0.975, df=n - 1) * np.std(data, ddof=1) / np.sqrt(n)
        self.stream.write(f'{name}\n')
        self.stream.write(f'Доверительный интервал для оценки \nмат. ожидания при неизвестном СКО (уровень доверия = 0.95): \n({left_edge:.3}, {right_edge:.3})\n\n')

    def disp_conf_interval_unknown_mean(self, data, name):
        n = len(data)
        t1 = ss.chi2.ppf(q=1-0.975, df=n-1)
        t2 = ss.chi2.ppf(q=1-0.025, df=n-1)
        left_edge = (n - 1) * np.var(data, ddof=1) / t2
        right_edge = (n - 1) * np.var(data, ddof=1) / t1
        self.stream.write(f'{name}\n')
        self.stream.write(f'Доверительный интервал для оценки \nдисперсии при неизвестном значении генерального среднего (уровень доверия = 0.95): \n[{left_edge:.3}, {right_edge:.3}]\n\n')

    def general_sd_conf_interval(self, data, name):
        n = len(data)
        t1 = ss.chi2.ppf(q=1-0.975, df=n-1)
        t2 = ss.chi2.ppf(q=1-0.025, df=n-1)
        left_edge = np.sqrt(n - 1) * np.std(data, ddof=1) / np.sqrt(t2)
        right_edge = np.sqrt(n - 1) * np.std(data, ddof=1) / np.sqrt(t1)
        self.stream.write(f'{name}\n')
        self.stream.write(f'Доверительный интервал для оценки \nгенерального СКО (уровень доверия = 0.95): \n[{left_edge:.3}, {right_edge:.3}]\n\n')

    def bernoulli(self):
        p_cur = 0.3
        sample = ss.bernoulli.rvs(p=p_cur, random_state=self.rs, size=500)
        n = len(sample)
        p = np.sum(sample) / n
        z = ss.norm.ppf(q=0.975, loc=0, scale=1)
        left_edge = p - z * np.sqrt(p * (1 - p)) / np.sqrt(n)
        right_edge = p + z * np.sqrt(p * (1 - p)) / np.sqrt(n)
        self.stream.write('Распределение Бернулли\n')
        self.stream.write(f'p = {p_cur}, q = {1 - p_cur}\n')
        self.stream.write(f'Доверительный интервал для оценки \nпараметра p в распределении Бернулли (уровень доверия = 0.95): \n({left_edge:.3}, {right_edge:.3})\n\n')


file = open(RESULT_ROOT / 'file.txt', 'w', encoding='utf-8')
Lab02(file).processing()
file.close()
